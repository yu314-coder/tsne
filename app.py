"""
t-SNE Explorer - Streamlit Application
A transparent t-SNE implementation with synthetic data generation and file upload support

This version uses Streamlit for the UI while maintaining the same functionality as app.py.
Backend is MCP-ready for future Android app integration.
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from io import BytesIO
from PIL import Image
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

try:
    from sklearn.manifold import TSNE as SklearnTSNE
    from sklearn.datasets import fetch_openml
except Exception:
    SklearnTSNE = None
    fetch_openml = None


# ==================== Styling ====================

def inject_custom_css():
    """Inject custom CSS from web/style.css to match the original design"""
    st.markdown("""
    <style>
    /* Import styling from web folder */
    :root {
        --primary: #667eea;
        --primary-dark: #5568d3;
        --secondary: #764ba2;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
    }

    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Headers */
    h1 {
        color: white;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    h2, h3 {
        color: #667eea;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }

    /* Dataframes */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Cards */
    .element-container {
        background: white;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== TSNEExplorer Backend Class (MCP-Ready) ====================

class TSNEExplorer:
    """
    Backend API for t-SNE computations.
    This class is MCP-ready - all methods return JSON-serializable data
    and can be called directly (Streamlit) or via API endpoints (future Android app).
    """

    def __init__(self):
        pass

    # ==================== Synthetic Data Generation ====================

    def generate_simplex_points(self, n, d, k, seed=42):
        """Generate n points in d dimensions with k distinct distance types"""
        np.random.seed(seed)

        # Validate inputs
        max_distances = (n * (n - 1)) // 2
        if k > max_distances:
            return {
                'success': False,
                'error': f'Cannot create {k} distinct distances with only {n} points. '
                        f'Maximum possible is {max_distances} distinct distances.'
            }

        if k < 1:
            return {
                'success': False,
                'error': f'k must be at least 1 (you specified k={k}).'
            }

        # Special case: k=1
        if k == 1:
            if n > d + 1:
                return {
                    'success': False,
                    'error': f'For k=1 (equidistant points), maximum n is {d+1} in {d}D.'
                }
            X = self._generate_regular_simplex(n, d)
        else:
            X = self._generate_k_distance_set(n, d, k, seed)

        # Compute pairwise distances
        distances = self._compute_pairwise_distances(X)
        unique_distances = np.unique(np.round(distances[distances > 0], decimals=6))

        return {
            'success': True,
            'points': X.tolist(),
            'n': n,
            'd': d,
            'k': k,
            'actual_k': len(unique_distances),
            'unique_distances': unique_distances.tolist(),
            'distances_min': float(np.min(distances[distances > 0])) if n > 1 else 0,
            'distances_mean': float(np.mean(distances[distances > 0])) if n > 1 else 0,
            'distances_max': float(np.max(distances)),
        }

    def _generate_regular_simplex(self, n, d):
        """Generate regular n-simplex with equal pairwise distances"""
        if n == 1:
            return np.zeros((1, d))

        if n == 2:
            X = np.zeros((2, d))
            X[0, 0] = -0.5
            X[1, 0] = 0.5
            return X

        vertices = np.eye(n)
        vertices = vertices - np.mean(vertices, axis=0)
        vertices = vertices / np.sqrt(2)

        if d >= n - 1:
            X = vertices[:, :min(d, n)]
            if d > n:
                X = np.pad(X, ((0, 0), (0, d - n)), 'constant')
        else:
            X = vertices[:, :d]

        return X

    def _generate_k_distance_set(self, n, d, k, seed):
        """Generate points aiming for k distinct pairwise distances"""
        np.random.seed(seed)

        if n <= 0 or d <= 0:
            return np.zeros((0, max(d, 0)))

        if n == 1:
            return np.zeros((1, d))

        # Exact k=2 constructions
        if k == 2:
            if d >= 2 and n == 5:
                return self._regular_ngon(n=5, d=d)
            if n <= 2 * d:
                return self._cross_polytope(n=n, d=d)
            return self._optimize_k_distance_set(n=n, d=d, k=k, seed=seed)

        # Exact k>=3 constructions
        if k >= 3 and d >= k and k <= 12 and n <= (2 ** k):
            return self._k_cube_k_distance_set(n=n, d=d, k=k)

        if k == 3 and d >= 2 and n in (6, 7):
            return self._regular_ngon(n=n, d=d)

        return self._optimize_k_distance_set(n=n, d=d, k=k, seed=seed)

    def _regular_ngon(self, n, d):
        """Regular n-gon in 2D"""
        X = np.zeros((n, d))
        if d < 2:
            return X
        angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
        X[:, 0] = np.cos(angles)
        X[:, 1] = np.sin(angles)
        return X

    def _cross_polytope(self, n, d):
        """Cross polytope vertices"""
        X = np.zeros((n, d))
        if n == 1:
            return X
        point_idx = 0
        for i in range(d):
            if point_idx >= n:
                break
            X[point_idx, i] = 1.0
            point_idx += 1
            if point_idx >= n:
                break
            X[point_idx, i] = -1.0
            point_idx += 1
        return X

    def _k_cube_k_distance_set(self, n, d, k):
        """k-dimensional hypercube vertices"""
        vertices = []
        seen = set()

        origin = tuple([0] * k)
        vertices.append(origin)
        seen.add(origin)

        for weight in range(1, k + 1):
            if len(vertices) >= n:
                break
            v = tuple([1] * weight + [0] * (k - weight))
            if v not in seen:
                vertices.append(v)
                seen.add(v)

        for mask in range(1, 2 ** k):
            if len(vertices) >= n:
                break
            v = tuple((mask >> bit) & 1 for bit in range(k))
            if v in seen:
                continue
            vertices.append(v)
            seen.add(v)

        Xk = np.array(vertices[:n], dtype=float)
        X = np.zeros((n, d), dtype=float)
        X[:, :k] = Xk
        X = X - X.mean(axis=0, keepdims=True)
        return X

    def _optimize_k_distance_set(self, n, d, k, seed, n_iter=2000, lr=0.02):
        """Heuristic optimization for k distances"""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d)) * 0.1

        if n < 2:
            return X

        D0 = self._compute_pairwise_distances(X)
        upper = D0[np.triu_indices(n, k=1)]
        if upper.size == 0:
            return X

        r_min = float(np.percentile(upper, 10))
        r_max = float(np.percentile(upper, 90))
        if r_max <= 1e-8:
            r_max = 1.0
        radii = np.linspace(max(r_min, 1e-3), max(r_max, 1e-3), k)

        use_minibatch = n > 150
        batch_size = min(5000, (n * (n - 1)) // 2) if use_minibatch else 0
        ema = 0.15

        for _ in range(n_iter):
            if use_minibatch:
                ii = rng.integers(0, n, size=batch_size)
                jj = rng.integers(0, n, size=batch_size)
                mask = ii != jj
                if not np.any(mask):
                    continue
                ii = ii[mask]
                jj = jj[mask]

                diff = X[ii] - X[jj]
                dist = np.sqrt(np.sum(diff * diff, axis=1))
                dist_safe = np.maximum(dist, 1e-12)

                assign = np.argmin(np.abs(dist[:, np.newaxis] - radii[np.newaxis, :]), axis=1)
                target = radii[assign]

                for m in range(k):
                    m_mask = assign == m
                    if np.any(m_mask):
                        radii[m] = (1 - ema) * radii[m] + ema * float(np.mean(dist[m_mask]))

                err = dist_safe - target
                coef = (2.0 * err / dist_safe)[:, np.newaxis]
                grad_pairs = coef * diff

                grad = np.zeros_like(X)
                np.add.at(grad, ii, grad_pairs)
                np.add.at(grad, jj, -grad_pairs)
            else:
                D = self._compute_pairwise_distances(X)
                iu, ju = np.triu_indices(n, k=1)
                dist = D[iu, ju]
                dist_safe = np.maximum(dist, 1e-12)

                assign = np.argmin(np.abs(dist[:, np.newaxis] - radii[np.newaxis, :]), axis=1)
                target = radii[assign]

                for m in range(k):
                    m_mask = assign == m
                    if np.any(m_mask):
                        radii[m] = float(np.mean(dist[m_mask]))

                err = dist_safe - target
                coef = (2.0 * err / dist_safe)[:, np.newaxis]
                diff = X[iu] - X[ju]
                grad_pairs = coef * diff

                grad = np.zeros_like(X)
                np.add.at(grad, iu, grad_pairs)
                np.add.at(grad, ju, -grad_pairs)

            grad += 1e-3 * X
            X = X - lr * grad
            X = X - X.mean(axis=0, keepdims=True)

        return X

    def _compute_pairwise_distances(self, X):
        """Compute pairwise Euclidean distances"""
        n = X.shape[0]
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    # ==================== MNIST Dataset ====================

    def load_mnist(self, max_samples=1000, subset='train'):
        """Load MNIST dataset"""
        try:
            if fetch_openml is None:
                return {'success': False, 'error': 'scikit-learn not available'}

            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

            all_images = np.array(mnist.data, dtype=np.float32)

            if isinstance(mnist.target[0], str):
                all_labels = np.array([int(label) for label in mnist.target], dtype=np.int32)
            else:
                all_labels = np.array(mnist.target, dtype=np.int32)

            if subset == 'train':
                images_flat = all_images[:60000]
                labels = all_labels[:60000]
            else:
                images_flat = all_images[60000:]
                labels = all_labels[60000:]

            if max_samples > 0 and max_samples < len(images_flat):
                images_flat = images_flat[:max_samples]
                labels = labels[:max_samples]

            X = images_flat / 255.0

            return {
                'success': True,
                'X': X,
                'labels': labels,
                'count': len(images_flat),
                'shape': X.shape,
                'message': f'Loaded {len(images_flat)} MNIST {subset} samples'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ==================== t-SNE Implementation ====================

    def run_tsne(self, X, perplexity=30, learning_rate=200, n_iter=1000,
                 early_exaggeration=12, momentum=0.8, seed=42, progress_callback=None):
        """Run t-SNE with transparent internals"""
        try:
            n, d = X.shape

            if n > 1000:
                return {'success': False, 'error': f'Dataset too large ({n} points). Please use n <= 1000.'}

            # Initialize Y
            np.random.seed(seed)
            Y = np.random.randn(n, 2) * 0.0001

            # Compute P
            if progress_callback:
                progress_callback(0, 'Computing P matrix...')
            P = self._compute_P(X, perplexity)

            # Optimize
            if progress_callback:
                progress_callback(0, 'Starting t-SNE optimization...')
            Y, Q, C_history = self._optimize_tsne(
                P, Y, learning_rate, n_iter, early_exaggeration, momentum, progress_callback
            )

            return {
                'success': True,
                'Y': Y,
                'P': P,
                'Q': Q,
                'C_history': C_history,
                'n': n
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _compute_P(self, X, perplexity):
        """Compute pairwise affinities P_ij"""
        n = X.shape[0]

        sum_X = np.sum(X**2, axis=1)
        D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * X @ X.T
        D = np.maximum(D, 0)

        P = np.zeros((n, n))
        target_entropy = np.log2(perplexity)

        for i in range(n):
            beta_min = -np.inf
            beta_max = np.inf
            beta = 1.0

            for _ in range(50):
                Di = D[i].copy()
                Di[i] = 0

                P_i = np.exp(-Di * beta)
                P_i[i] = 0
                sum_P_i = np.sum(P_i)

                if sum_P_i == 0:
                    P_i = np.ones(n) / n
                    sum_P_i = 1.0

                P_i = P_i / sum_P_i

                P_i_nonzero = P_i[P_i > 1e-12]
                H = -np.sum(P_i_nonzero * np.log2(P_i_nonzero))

                H_diff = H - target_entropy
                if np.abs(H_diff) < 1e-5:
                    break

                if H_diff > 0:
                    beta_min = beta
                    if beta_max == np.inf:
                        beta = beta * 2
                    else:
                        beta = (beta + beta_max) / 2
                else:
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta = beta / 2
                    else:
                        beta = (beta + beta_min) / 2

            P[i] = P_i

        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)

        return P

    def _optimize_tsne(self, P, Y, learning_rate, n_iter, early_exaggeration, momentum, progress_callback=None):
        """Optimize t-SNE using gradient descent"""
        n = Y.shape[0]
        Y_velocity = np.zeros_like(Y)
        C_history = []

        P_exag = P * early_exaggeration

        for iteration in range(n_iter):
            P_current = P_exag if iteration < 250 else P

            sum_Y = np.sum(Y**2, axis=1)
            D_low = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * Y @ Y.T
            D_low = np.maximum(D_low, 0)

            Q = (1 + D_low) ** (-1)
            np.fill_diagonal(Q, 0)
            sum_Q = np.sum(Q)
            if sum_Q < 1e-12:
                sum_Q = 1e-12
            Q = Q / sum_Q
            Q = np.maximum(Q, 1e-12)

            C = np.sum(P_current * np.log((P_current + 1e-12) / (Q + 1e-12)))
            C_history.append(float(C))

            PQ_diff = P_current - Q
            repulsion = (1 + D_low) ** (-1)
            attraction_repulsion = (PQ_diff * repulsion)[:, :, np.newaxis]
            Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
            gradient = 4 * (attraction_repulsion * Y_diff).sum(axis=1)

            Y_velocity = momentum * Y_velocity - learning_rate * gradient
            Y = Y + Y_velocity
            Y = Y - Y.mean(axis=0)

            if progress_callback and iteration % 10 == 0:
                progress_callback(iteration / n_iter, f'Iteration {iteration}/{n_iter}, Cost: {C:.4f}')

        # Final Q computation
        sum_Y = np.sum(Y**2, axis=1)
        D_low = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * Y @ Y.T
        D_low = np.maximum(D_low, 0)
        Q = (1 + D_low) ** (-1)
        np.fill_diagonal(Q, 0)
        sum_Q = np.sum(Q)
        if sum_Q < 1e-12:
            sum_Q = 1e-12
        Q = Q / sum_Q
        Q = np.maximum(Q, 1e-12)

        if progress_callback:
            progress_callback(1.0, 'Complete!')

        return Y, Q, C_history

    # ==================== Clustering ====================

    def run_clustering(self, Y, method='kmeans', k=3, eps=0.5, min_samples=5):
        """Run clustering on t-SNE results"""
        try:
            if method == 'kmeans':
                labels = self._kmeans(Y, k)
            elif method == 'dbscan':
                labels = self._dbscan(Y, eps, min_samples)
            else:
                return {'success': False, 'error': 'Unknown clustering method'}

            unique_labels = np.unique(labels)
            summary = []
            for label in unique_labels:
                count = np.sum(labels == label)
                summary.append({
                    'label': int(label),
                    'count': int(count)
                })

            return {
                'success': True,
                'labels': labels.tolist(),
                'summary': summary
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _kmeans(self, X, k, max_iter=100):
        """K-means clustering"""
        n = X.shape[0]
        indices = np.random.choice(n, k, replace=False)
        centroids = X[indices].copy()
        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            distances = np.zeros((n, k))
            for i in range(k):
                distances[:, i] = np.sum((X - centroids[i])**2, axis=1)

            new_labels = np.argmin(distances, axis=1)

            if np.all(labels == new_labels):
                break

            labels = new_labels

            for i in range(k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    centroids[i] = cluster_points.mean(axis=0)

        return labels

    def _dbscan(self, X, eps, min_samples):
        """DBSCAN clustering"""
        n = X.shape[0]
        labels = -np.ones(n, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue

            neighbors = self._find_neighbors(X, i, eps)

            if len(neighbors) < min_samples:
                labels[i] = -1
            else:
                self._expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_samples)
                cluster_id += 1

        return labels

    def _find_neighbors(self, X, point_idx, eps):
        """Find neighbors within eps distance"""
        distances = np.sum((X - X[point_idx])**2, axis=1)
        return np.where(distances <= eps**2)[0]

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id, eps, min_samples):
        """Expand cluster from seed point"""
        labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            if labels[neighbor_idx] != -1:
                i += 1
                continue

            labels[neighbor_idx] = cluster_id

            new_neighbors = self._find_neighbors(X, neighbor_idx, eps)
            if len(new_neighbors) >= min_samples:
                neighbors = np.concatenate([neighbors, new_neighbors])

            i += 1


# ==================== Streamlit UI ====================

def main():
    # Page config
    st.set_page_config(
        page_title="t-SNE Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject custom CSS
    inject_custom_css()

    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 40px; text-align: center; border-radius: 16px; margin-bottom: 20px;">
        <h1 style="color: white; font-size: 3em; margin-bottom: 10px;">t-SNE Explorer</h1>
        <p style="color: white; font-size: 1.2em; opacity: 0.95;">
            Transparent t-SNE with synthetic data generation and file uploads
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize backend
    if 'backend' not in st.session_state:
        st.session_state.backend = TSNEExplorer()

    # Initialize session state
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None

    # Sidebar navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Select Section", ["t-SNE", "Upload"])

    if tab == "t-SNE":
        tsne_tab()
    else:
        upload_tab()


def tsne_tab():
    """Main t-SNE tab"""
    st.header("t-SNE Analysis")

    # Section A: Synthetic Data Generator
    with st.expander("A) Synthetic Data Generator", expanded=True):
        st.info("Generate n points in d dimensions with k distinct distance types. "
                "Optimal: k=1 (n≤d+1 simplex), k=2 (n=5 pentagon), k=3 (n=7 heptagon).")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n = st.number_input("n (points)", min_value=1, max_value=100, value=6)
        with col2:
            d = st.number_input("d (dimensions)", min_value=1, max_value=100, value=10)
        with col3:
            k = st.number_input("k (distance types)", min_value=1, value=2)
        with col4:
            seed = st.number_input("seed", min_value=0, value=42)

        if st.button("Generate Points", key="gen_points"):
            with st.spinner("Generating synthetic data..."):
                result = st.session_state.backend.generate_simplex_points(n, d, k, seed)

                if result['success']:
                    # Store dataset
                    dataset_id = f"synthetic_{len(st.session_state.datasets)}"
                    st.session_state.datasets[dataset_id] = {
                        'type': 'synthetic',
                        'X': np.array(result['points']),
                        'shape': (result['n'], result['d'])
                    }

                    # Display stats
                    st.success(f"Generated {result['n']} points successfully!")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Points", result['n'])
                    col2.metric("Dimensions", result['d'])
                    col3.metric("Target k", result['k'])
                    col4.metric("Actual k", result['actual_k'])

                    st.write(f"**Unique Distances:** {', '.join([f'{d:.4f}' for d in result['unique_distances']])}")
                    st.write(f"**Range:** min={result['distances_min']:.4f}, "
                            f"mean={result['distances_mean']:.4f}, max={result['distances_max']:.4f}")

                    # Display points table
                    points_df = pd.DataFrame(
                        result['points'],
                        columns=[f'x{i+1}' for i in range(result['d'])]
                    )
                    st.dataframe(points_df.head(10), use_container_width=True)
                else:
                    st.error(result['error'])

    # Section B: MNIST Loader
    with st.expander("B) Load MNIST Dataset"):
        col1, col2 = st.columns(2)
        with col1:
            subset = st.selectbox("Subset", ["train", "test"])
        with col2:
            max_samples = st.number_input("Samples", min_value=100, max_value=10000, value=1000, step=100)

        if st.button("Load MNIST", key="load_mnist"):
            with st.spinner("Loading MNIST dataset..."):
                progress_bar = st.progress(0)
                progress_bar.progress(0.3)

                result = st.session_state.backend.load_mnist(max_samples, subset)
                progress_bar.progress(1.0)

                if result['success']:
                    dataset_id = f"mnist_{len(st.session_state.datasets)}"
                    st.session_state.datasets[dataset_id] = {
                        'type': 'mnist',
                        'X': result['X'],
                        'labels': result['labels'],
                        'count': result['count']
                    }
                    st.success(result['message'])
                else:
                    st.error(result['error'])

    # Section C: t-SNE Runner
    with st.expander("C) t-SNE Runner", expanded=True):
        # Dataset selector
        dataset_options = {f"{k} ({v['type']})": k for k, v in st.session_state.datasets.items()}

        if len(dataset_options) == 0:
            st.warning("No datasets available. Generate synthetic data or load MNIST first.")
            return

        selected_dataset_key = st.selectbox(
            "Select Dataset",
            options=list(dataset_options.keys())
        )
        selected_dataset_id = dataset_options[selected_dataset_key]

        # t-SNE parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            perplexity = st.number_input("Perplexity", min_value=5, max_value=50, value=30)
            learning_rate = st.number_input("Learning Rate", min_value=10, max_value=1000, value=200)
        with col2:
            iterations = st.number_input("Iterations", min_value=100, max_value=5000, value=1000)
            early_exag = st.number_input("Early Exaggeration", min_value=1, max_value=50, value=12)
        with col3:
            momentum = st.number_input("Momentum", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
            tsne_seed = st.number_input("Seed", min_value=0, value=42, key="tsne_seed")

        if st.button("Run t-SNE", key="run_tsne"):
            dataset = st.session_state.datasets[selected_dataset_id]
            X = dataset['X']

            progress_bar = st.progress(0)
            progress_text = st.empty()

            def progress_callback(progress, message):
                progress_bar.progress(progress)
                progress_text.text(message)

            result = st.session_state.backend.run_tsne(
                X, perplexity, learning_rate, iterations,
                early_exag, momentum, tsne_seed, progress_callback
            )

            if result['success']:
                st.session_state.current_results = result
                st.session_state.current_results['dataset_id'] = selected_dataset_id
                st.session_state.current_results['labels'] = dataset.get('labels')
                st.success("t-SNE completed successfully!")
                st.rerun()
            else:
                st.error(result['error'])

    # Section D: Results Display
    if st.session_state.current_results:
        display_results()


def display_results():
    """Display t-SNE results"""
    st.header("Results & Internals")

    results = st.session_state.current_results
    Y = np.array(results['Y'])
    P = np.array(results['P'])
    Q = np.array(results['Q'])
    C_history = results['C_history']
    labels = results.get('labels')

    # 2D Scatter Plot
    st.subheader("2D t-SNE Embedding")

    if labels is not None:
        # Color by labels
        fig = go.Figure()

        unique_labels = np.unique(labels)
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                  '#1abc9c', '#e67e22', '#95a5a6', '#34495e', '#c0392b']

        for label in unique_labels:
            mask = labels == label
            fig.add_trace(go.Scatter(
                x=Y[mask, 0],
                y=Y[mask, 1],
                mode='markers',
                name=f'Digit {label}',
                marker=dict(size=8, color=colors[int(label) % len(colors)],
                           line=dict(color='white', width=1))
            ))
    else:
        # Default plot
        fig = go.Figure(data=go.Scatter(
            x=Y[:, 0],
            y=Y[:, 1],
            mode='markers+text',
            text=[f'y{i+1}' for i in range(len(Y))],
            textposition='top center',
            marker=dict(size=10, color='#667eea', line=dict(color='white', width=1))
        ))

    fig.update_layout(
        title="t-SNE Embedding",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cost Plot
    st.subheader("Cost (KL Divergence) Over Iterations")
    fig_cost = go.Figure(data=go.Scatter(
        y=C_history,
        mode='lines',
        line=dict(color='#e74c3c', width=2)
    ))
    fig_cost.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Cost (KL Divergence)",
        height=400
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    # Matrices
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("P Matrix (High-D Affinities)")
        fig_p = go.Figure(data=go.Heatmap(z=P, colorscale='Viridis'))
        fig_p.update_layout(height=400)
        st.plotly_chart(fig_p, use_container_width=True)

    with col2:
        st.subheader("Q Matrix (Low-D Affinities)")
        fig_q = go.Figure(data=go.Heatmap(z=Q, colorscale='Viridis'))
        fig_q.update_layout(height=400)
        st.plotly_chart(fig_q, use_container_width=True)

    # Coordinates Table
    st.subheader("2D Coordinates")
    coords_df = pd.DataFrame(Y, columns=['Dim 1', 'Dim 2'])
    coords_df.index = [f'y{i+1}' for i in range(len(Y))]
    st.dataframe(coords_df.head(20), use_container_width=True)

    # Export
    if st.button("Export Results (CSV)"):
        csv = coords_df.to_csv()
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="tsne_results.csv",
            mime="text/csv"
        )
        st.success("Results exported!")

    # Clustering section
    if labels is not None:
        st.subheader("Clustering")
        cluster_method = st.selectbox("Method", ["kmeans", "dbscan"])

        if cluster_method == "kmeans":
            k = st.number_input("k (clusters)", min_value=2, max_value=10, value=3)
            if st.button("Run K-Means"):
                result = st.session_state.backend.run_clustering(Y, 'kmeans', k=k)
                if result['success']:
                    st.success("Clustering complete!")
                    st.write("**Cluster Summary:**")
                    st.json(result['summary'])
        else:
            col1, col2 = st.columns(2)
            with col1:
                eps = st.number_input("eps", min_value=0.1, value=0.5, step=0.1)
            with col2:
                min_samples = st.number_input("min_samples", min_value=1, value=5)
            if st.button("Run DBSCAN"):
                result = st.session_state.backend.run_clustering(Y, 'dbscan', eps=eps, min_samples=min_samples)
                if result['success']:
                    st.success("Clustering complete!")
                    st.write("**Cluster Summary:**")
                    st.json(result['summary'])


def upload_tab():
    """Upload tab for CSV and images"""
    st.header("Upload Data")

    st.subheader("CSV Files")
    uploaded_csv = st.file_uploader("Upload CSV", type=['csv'], accept_multiple_files=False)

    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)

            dataset_id = f"csv_{len(st.session_state.datasets)}"
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            st.success(f"Uploaded {uploaded_csv.name}")
            st.write(f"Shape: {df.shape}")
            st.write(f"Numeric columns: {', '.join(numeric_columns)}")

            st.dataframe(df.head())

            # Column selector
            selected_cols = st.multiselect("Select numeric columns", numeric_columns, default=numeric_columns)
            handle_missing = st.selectbox("Handle missing values", ["drop", "mean", "zero"])

            if st.button("Prepare Dataset"):
                if selected_cols:
                    df_subset = df[selected_cols]

                    if handle_missing == 'drop':
                        df_subset = df_subset.dropna()
                    elif handle_missing == 'mean':
                        df_subset = df_subset.fillna(df_subset.mean())
                    elif handle_missing == 'zero':
                        df_subset = df_subset.fillna(0)

                    X = df_subset.values

                    st.session_state.datasets[dataset_id] = {
                        'type': 'csv',
                        'X': X,
                        'shape': X.shape,
                        'name': uploaded_csv.name
                    }

                    st.success(f"Dataset prepared: {X.shape[0]} rows × {X.shape[1]} columns")
                else:
                    st.warning("Please select at least one column")

        except Exception as e:
            st.error(f"Error uploading CSV: {str(e)}")

    # Dataset list
    st.subheader("Uploaded Datasets")
    if len(st.session_state.datasets) == 0:
        st.info("No datasets uploaded yet")
    else:
        for dataset_id, dataset in st.session_state.datasets.items():
            if dataset['type'] == 'synthetic':
                st.write(f"🔢 Synthetic: {dataset['shape'][0]}×{dataset['shape'][1]}")
            elif dataset['type'] == 'csv':
                st.write(f"📊 CSV: {dataset.get('name', 'Unknown')} ({dataset['shape'][0]}×{dataset['shape'][1]})")
            elif dataset['type'] == 'mnist':
                st.write(f"✏️ MNIST: {dataset['count']} samples")


if __name__ == '__main__':
    main()
