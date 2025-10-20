import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.manifold import TSNE
import plotly.graph_objects as go

st.set_page_config(page_title="Interactive t-SNE", layout="wide")
st.title("t-SNE Projection to 2 Dimensions")

st.write(
    "Generate n points with equal pairwise distances in R^n dimensional space (regular simplex) and project them to 2D using t-SNE."
)

n_points = st.slider("Number of points (n)", min_value=3, max_value=500, value=4, step=1)

point_generation_mode = st.radio(
    "Point generation method",
    ("Regular simplex (all distances equal)", "Custom: x₁...x_(n-1) distance=1, x_n distance=d to all others", "Custom high-dimensional input points"),
    index=0,
)

custom_distance = None
custom_input_text = None

if point_generation_mode == "Custom: x₁...x_(n-1) distance=1, x_n distance=d to all others":
    custom_distance = st.number_input("Distance d from x_n to other points", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
elif point_generation_mode == "Custom high-dimensional input points":
    custom_input_example = "1,0,0,0\n0,1,0,0\n0,0,1,0\n0,0,0,1"
    custom_input_text = st.text_area(
        "Custom high-dimensional input points (x)",
        value=custom_input_example,
        height=200,
        help="Provide comma- or tab-separated values per row. Each row is one point x_i. Number of rows determines n.",
        key="custom_input_points",
    )

initialization_mode = st.radio(
    "Initialization",
    ("Random (default)", "Provide custom 2D starting coordinates"),
    index=0,
)

custom_init_text = None
if initialization_mode == "Provide custom 2D starting coordinates":
    custom_init_example = "0,0\n1,0\n0.5,0.866\n0.5,0.289"
    custom_init_text = st.text_area(
        "Custom initial 2D coordinates (y₀)",
        value=custom_init_example,
        height=120,
        help="Provide two comma- or tab-separated values per row. Number of rows must match n.",
        key="custom_init_input",
    )

use_custom_sigma = st.checkbox("Use custom σ (ignore perplexity)", value=False, help="When checked, uses a custom sigma value for all points instead of tuning it to match perplexity")

custom_sigma = 2**(-0.5)
if use_custom_sigma:
    custom_sigma = st.number_input("Custom σ value", min_value=0.0, max_value=100.0, value=2**(-0.5), format="%.10f", step=0.01, help="Default is 2^(-0.5) ≈ 0.7071067812")

col1, col2 = st.columns(2)
with col1:
    # Default perplexity should be at most n-1
    default_perp = min(3, max(1, n_points - 1))
    perplexity = st.slider("Perplexity", min_value=1, max_value=100, value=default_perp, step=1, disabled=use_custom_sigma)
    if perplexity >= n_points and not use_custom_sigma:
        st.warning(f"⚠️ Perplexity ({perplexity}) should be less than n ({n_points}). Maximum meaningful perplexity is {n_points-1}.")
    learning_rate = st.number_input("Learning rate", min_value=0.0, max_value=1000.0, value=0.0001, format="%.25f", step=0.00001)
with col2:
    iterations = st.slider("Number of iterations", min_value=250, max_value=2000000, value=1000, step=50)
    metric = st.selectbox("Distance metric", ["euclidean", "cosine", "manhattan"])

run_tsne = st.button("Compute t-SNE")

@st.cache_data(show_spinner=False)
def parse_input(text: str) -> pd.DataFrame:
    data = io.StringIO(text.strip())
    try:
        df = pd.read_csv(data, sep=None, engine="python", header=None)
    except pd.errors.ParserError:
        df = pd.read_csv(io.StringIO(text.strip()), sep=r"\s+", header=None)
    if df.empty:
        raise ValueError("No data found.")
    if df.shape[0] < 2:
        raise ValueError("Need at least two points.")
    return df

@st.cache_data(show_spinner=False)
def generate_equidistant_points(n: int) -> pd.DataFrame:
    """Generate n points where all pairwise distances are exactly equal (regular simplex)."""
    # Simple construction: use standard basis vectors in R^n, center them
    # This creates a regular simplex with all pairwise distances equal

    points = np.eye(n)  # n points in R^n (standard basis vectors)
    points = points - np.mean(points, axis=0)  # Center at origin

    # Scale so all pairwise distances equal 1
    dist = np.linalg.norm(points[0] - points[1])
    if dist > 0:
        points = points / dist

    return pd.DataFrame(points)

@st.cache_data(show_spinner=False)
def generate_custom_distance_points(n: int, d_custom: float) -> pd.DataFrame:
    """Generate n points where x_1...x_(n-1) have distance 1, and x_n has distance d to all others."""
    if n < 2:
        raise ValueError("Need at least 2 points")

    if n == 2:
        # Special case: just two points at distance d_custom
        points = np.zeros((2, 2))
        points[0] = [0.0, 0.0]
        points[1] = [d_custom, 0.0]
        return pd.DataFrame(points)

    # Generate first n-1 points forming regular simplex with distance 1
    m = n - 1
    simplex_points = np.eye(m)  # m points in R^m
    simplex_points = simplex_points - np.mean(simplex_points, axis=0)  # Center

    # Scale to distance 1
    edge_dist = np.linalg.norm(simplex_points[0] - simplex_points[1])
    if edge_dist > 0:
        simplex_points = simplex_points / edge_dist

    # Calculate centroid (should be zero, but calculate for safety)
    centroid = np.mean(simplex_points, axis=0)

    # Calculate distance from centroid to any vertex
    r = np.linalg.norm(simplex_points[0] - centroid)

    # Place x_n at distance d_custom from all simplex vertices
    # Use one additional dimension for placement
    # Pythagorean theorem: d_custom^2 = r^2 + h^2
    if d_custom**2 < r**2:
        h = 0  # Best approximation if d_custom too small
    else:
        h = np.sqrt(d_custom**2 - r**2)

    # Embed in R^(m+1) = R^n
    full_points = np.zeros((n, n))
    full_points[:m, :m] = simplex_points  # First n-1 points
    full_points[m, :m] = centroid  # x_n at centroid in first m dimensions
    full_points[m, m] = h  # Offset in new dimension

    return pd.DataFrame(full_points)

@st.cache_data(show_spinner=True)
def compute_tsne(
    df: pd.DataFrame,
    perplexity: float,
    learning_rate: float,
    max_iter: int,
    metric: str,
    init_mode: str,
    init_values: tuple | None,
) -> pd.DataFrame:
    if init_mode == "custom":
        if init_values is None:
            raise ValueError("Custom initialization values were not provided.")
        init_array = np.asarray(init_values, dtype=np.float64)
        if init_array.shape != (df.shape[0], 2):
            raise ValueError("Custom initialization must match dataset size (rows) and have exactly two columns.")
        init_param = init_array
    else:
        init_param = "random"

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
        init=init_param,
        random_state=42,
    )
    embedding = tsne.fit_transform(df.values)
    return pd.DataFrame(embedding, columns=["x", "y"], index=df.index)

# Pairwise squared Euclidean distances (vectorized) for probability and gradient calculations.
def compute_squared_euclidean(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    sum_sq = np.sum(np.square(matrix), axis=1, keepdims=True)
    distances = sum_sq + sum_sq.T - 2.0 * matrix @ matrix.T
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, 0.0)
    return distances

# High-dimensional joint probabilities P_ij computed to match the chosen perplexity.
def compute_joint_probabilities(data: np.ndarray, perplexity: float, tol: float = 1e-5, max_iter: int = 50, custom_sigma: float | None = None) -> np.ndarray:
    distances = compute_squared_euclidean(data)
    n_samples = distances.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least two points to compute probabilities.")

    target_entropy = np.log(perplexity)
    P = np.zeros_like(distances)

    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        dist_i = distances[i, mask]

        if custom_sigma is not None:
            # Use custom sigma value, beta = 1/(2*sigma^2)
            beta = 1.0 / (2.0 * custom_sigma**2)
            p = np.exp(-dist_i * beta)
            sum_p = np.sum(p)
            if sum_p == 0.0:
                p = np.full_like(dist_i, 1.0 / dist_i.size)
            else:
                p /= sum_p
        else:
            # Binary search to find beta that matches target perplexity
            beta = 1.0
            beta_min = -np.inf
            beta_max = np.inf

            for _ in range(max_iter):
                p = np.exp(-dist_i * beta)
                sum_p = np.sum(p)
                if sum_p == 0.0:
                    p = np.full_like(dist_i, 1.0 / dist_i.size)
                    break

                H = np.log(sum_p) + (beta * np.sum(dist_i * p) / sum_p)
                H_diff = H - target_entropy

                if abs(H_diff) < tol:
                    p /= sum_p
                    break

                if H_diff > 0.0:
                    beta_min = beta
                    if np.isinf(beta_max):
                        beta *= 2.0
                    else:
                        beta = 0.5 * (beta + beta_max)
                else:
                    beta_max = beta
                    if np.isinf(beta_min):
                        beta *= 0.5
                    else:
                        beta = 0.5 * (beta + beta_min)
            else:
                p = np.exp(-dist_i * beta)
                sum_p = np.sum(p)
                if sum_p == 0.0:
                    p = np.full_like(dist_i, 1.0 / dist_i.size)
                else:
                    p /= sum_p

        P[i, mask] = p

    # Symmetric joint probability: p_ij = (p_j|i + p_i|j) / (2N)
    # This formula already ensures that sum of all p_ij = 1
    P = (P + P.T) / (2.0 * n_samples)
    np.fill_diagonal(P, 0.0)
    # Apply minimum threshold for numerical stability
    P = np.maximum(P, 1e-12)
    return P

# Low-dimensional affinities Q_ij and the unnormalized Student-t kernel weights.
def compute_low_dim_affinities(embedding: np.ndarray):
    distances = compute_squared_euclidean(embedding)
    # Student t-distribution: q_ij = (1 + ||y_i - y_j||²)^(-1) / Z
    inv_distances = 1.0 / (1.0 + distances)
    np.fill_diagonal(inv_distances, 0.0)
    # Normalize: sum over all i≠j
    denom = np.sum(inv_distances)
    denom = max(denom, 1e-12)
    Q = inv_distances / denom
    # Apply minimum threshold for numerical stability
    Q = np.maximum(Q, 1e-12)
    np.fill_diagonal(Q, 0.0)
    return Q, inv_distances

# Gradient of the Kullback-Leibler divergence with respect to each embedding point.
def compute_gradients(embedding: np.ndarray, P: np.ndarray, Q: np.ndarray, inv_distances: np.ndarray) -> np.ndarray:
    Y = np.asarray(embedding, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    inv_distances = np.asarray(inv_distances, dtype=np.float64)

    n_samples = Y.shape[0]
    gradients = np.zeros_like(Y)
    for i in range(n_samples):
        diff = Y[i] - Y
        coeff = (P[i] - Q[i]) * inv_distances[i]
        coeff[i] = 0.0
        gradients[i] = 4.0 * np.sum(coeff[:, None] * diff, axis=0)
    return gradients

def compute_kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute KL divergence between P and Q."""
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    return np.sum(P * np.log(P / Q))

if run_tsne:
    try:
        # Generate points based on selected method
        if point_generation_mode == "Custom: x₁...x_(n-1) distance=1, x_n distance=d to all others":
            data_df = generate_custom_distance_points(n_points, custom_distance)
            point_desc = f"Custom Distance Points in R^{n_points}"
        elif point_generation_mode == "Custom high-dimensional input points":
            data_df = parse_input(custom_input_text or "")
            if data_df.shape[0] < 2:
                raise ValueError("Need at least two points.")
            point_desc = f"Custom Input Points in R^{data_df.shape[1]}"
        else:
            data_df = generate_equidistant_points(n_points)
            point_desc = f"Generated Equidistant Points in R^{n_points}"

        if perplexity >= data_df.shape[0]:
            st.warning("Perplexity must be less than the number of samples.")
        else:
            # Display generated points
            st.subheader(point_desc)
            # Label points as x_1, x_2, ..., x_n and columns as coordinate dimensions (x, y, z, w, ...)
            data_display = data_df.copy()
            # Use x, y, z, w for first 4 dimensions, then x₅, x₆, ... for higher dimensions
            coord_names = ['x', 'y', 'z', 'w'] + [f'x_{i+1}' for i in range(4, data_df.shape[1])]
            data_display.columns = coord_names[:data_df.shape[1]]
            data_display.index = [f"x_{i+1}" for i in range(len(data_display))]
            st.dataframe(data_display)

            # Display distance matrix for verification
            distances = compute_squared_euclidean(data_df.values)
            distances = np.sqrt(distances)  # Convert to actual distances
            st.subheader("Pairwise Distances")
            dist_df = pd.DataFrame(distances,
                                   index=[f"x_{i+1}" for i in range(len(data_df))],
                                   columns=[f"x_{i+1}" for i in range(len(data_df))])
            st.dataframe(dist_df)

            # Compute original cost C (before t-SNE)
            sigma_to_use = custom_sigma if use_custom_sigma else None
            p_matrix_original = compute_joint_probabilities(data_df.values, perplexity, custom_sigma=sigma_to_use)

            if use_custom_sigma:
                beta_value = 1.0 / (2.0 * custom_sigma**2)
                st.info(f"σ is set to {custom_sigma:.10f} for all points (β = {beta_value:.10f}). Perplexity is ignored.")

            init_mode_key = "random"
            init_values_tuple: tuple | None = None

            if initialization_mode == "Provide custom 2D starting coordinates":
                custom_df = parse_input(custom_init_text or "")
                if custom_df.shape[1] != 2:
                    raise ValueError("Custom initial coordinates must have exactly two columns.")
                if custom_df.shape[0] != data_df.shape[0]:
                    raise ValueError("Custom initial coordinates must have the same number of rows as n.")
                custom_array = custom_df.to_numpy(dtype=np.float64)
                init_values_tuple = tuple(tuple(row) for row in custom_array)
                init_mode_key = "custom"

                # Compute initial cost with custom initialization
                q_matrix_init, _ = compute_low_dim_affinities(custom_array)
                cost_initial = compute_kl_divergence(p_matrix_original, q_matrix_init)
                st.info(f"Initial Cost C (with custom y₀): {cost_initial:.6f}")

            embedding_df = compute_tsne(
                data_df,
                perplexity,
                learning_rate,
                iterations,
                metric,
                init_mode_key,
                init_values_tuple,
            )

            # Compute final cost C (after t-SNE)
            q_matrix_final, _ = compute_low_dim_affinities(embedding_df.values)
            cost_final = compute_kl_divergence(p_matrix_original, q_matrix_final)

            st.success(f"Final Cost C (after t-SNE): {cost_final:.6f}")

            st.subheader("t-SNE Embedding (y)")
            embedding_display = embedding_df.copy()
            embedding_display.columns = ["x", "y"]
            embedding_display.index = [f"y_{i+1}" for i in range(len(embedding_display))]
            st.dataframe(embedding_display)

            # Display distance matrix for output y
            y_distances = compute_squared_euclidean(embedding_df.values)
            y_distances = np.sqrt(y_distances)  # Convert to actual distances
            st.subheader("Pairwise Distances in 2D Embedding (y)")
            y_dist_df = pd.DataFrame(y_distances,
                                     index=[f"y_{i+1}" for i in range(len(embedding_df))],
                                     columns=[f"y_{i+1}" for i in range(len(embedding_df))])
            st.dataframe(y_dist_df)

            # Plot 2D scatter plot with Plotly
            fig = go.Figure()

            # Add scatter points
            fig.add_trace(go.Scatter(
                x=embedding_df["x"],
                y=embedding_df["y"],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=list(range(len(embedding_df))),
                    colorscale='Viridis',
                    line=dict(color='black', width=2),
                    showscale=False
                ),
                text=[f'y_{i+1}' for i in range(len(embedding_df))],
                textposition="top center",
                textfont=dict(size=14, color='black', family='Arial Black'),
                hovertemplate='<b>y_%{pointNumber|add:1}</b><br>' +
                              'x: %{x:.6f}<br>' +
                              'y: %{y:.6f}<br>' +
                              '<extra></extra>',
                showlegend=False
            ))

            # Update layout for better appearance
            fig.update_layout(
                title=dict(
                    text="t-SNE 2D Embedding",
                    font=dict(size=24, color='black', family='Arial Black'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(text="x", font=dict(size=18, color='black', family='Arial')),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='black'
                ),
                yaxis=dict(
                    title=dict(text="y", font=dict(size=18, color='black', family='Arial')),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='black',
                    scaleanchor="x",
                    scaleratio=1
                ),
                width=900,
                height=800,
                plot_bgcolor='white',
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                label="Download embedding as CSV",
                data=embedding_df.to_csv(index=False).encode("utf-8"),
                file_name="tsne_embedding.csv",
                mime="text/csv",
            )

            # Display probability matrices
            st.subheader("High-dimensional joint probabilities P_ij")
            p_matrix_df = pd.DataFrame(p_matrix_original,
                                       index=[f"x_{i+1}" for i in range(len(data_df))],
                                       columns=[f"x_{i+1}" for i in range(len(data_df))])
            st.dataframe(p_matrix_df)

            try:
                q_matrix, inv_distances = compute_low_dim_affinities(embedding_df.values)

                st.subheader("Low-dimensional affinities Q_ij")
                q_matrix_df = pd.DataFrame(q_matrix,
                                           index=[f"x_{i+1}" for i in range(len(data_df))],
                                           columns=[f"x_{i+1}" for i in range(len(data_df))])
                st.dataframe(q_matrix_df)

                gradients = compute_gradients(embedding_df.values, p_matrix_original, q_matrix, inv_distances)
                gradient_df = pd.DataFrame(gradients, columns=["∂C/∂x", "∂C/∂y"])
                gradient_df.index = [f"y_{i+1}" for i in range(len(gradient_df))]

                st.subheader("Gradient ∂C/∂y")
                st.dataframe(gradient_df)
                st.download_button(
                    label="Download gradients as CSV",
                    data=gradient_df.to_csv(index=True).encode("utf-8"),
                    file_name="tsne_gradients.csv",
                    mime="text/csv",
                )
            except Exception as grad_exc:
                st.error(f"Could not compute gradient or probabilities: {grad_exc}")
    except Exception as exc:
        st.error(f"Could not compute t-SNE: {exc}")
