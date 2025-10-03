import streamlit as st
import numpy as np
import pandas as pd
from math import exp, log, pi, cos, sin
from numpy.linalg import norm
from tqdm import tqdm
import plotly.graph_objects as go

try:
    from scipy.optimize import least_squares
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="Solve t-SNE: dC/dy=0", layout="wide")

st.title("Two-distance t-SNE: solve $dC/dy_i=0$ from definitions")

st.markdown(
    'This app uses **only** the definitions of $p_{ij}$, $q_{ij}(Y)$, and the t-SNE gradient $dC/dy_i = 4\sum_{j\\neq i}(p_{ij}-q_{ij})\\frac{y_i-y_j}{1+||y_i-y_j||^2}$.'
)
st.markdown(
    'Fixed initialization: **n distinct points** $Y_0$ (regular $n$-gon with fixed jitter pattern). Solver finds Y where $dC/dy_i=0$ for all i (critical point of KL cost).'
)

with st.sidebar:
    st.header("Inputs")
    n = st.number_input("n (≥ 4)", min_value=4, value=8, step=1)

    # Configuration selection
    config_type = st.selectbox(
        "Distance configuration",
        ["Random pairs", "Star configuration"],
        help="Random: β fraction of pairs at distance a. Star: (n-k, k) split with k points at distance d from all others, remaining at distance 1"
    )

    if config_type == "Random pairs":
        beta = st.slider("β (fraction of unordered pairs at distance a)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        a = st.number_input("a (> 0)", min_value=1e-6, value=2.0, step=0.1, format="%.6f")
    else:  # Star configuration
        n_far = st.number_input("Number of points at distance d", min_value=1, max_value=int(n)-1, value=1, step=1)
        d_star = st.number_input("d (distance for far points, > 0)", min_value=1e-6, value=2.0, step=0.1, format="%.6f")
        n_close = int(n) - n_far
        st.info(f"Star config: ({n_close}, {n_far}) = {n_close} points at distance 1 from each other, {n_far} point(s) at distance {d_star} from all others")

    sigma = st.number_input("Gaussian σ for P", min_value=1e-9, value=1.0, step=0.1, format="%.6f")
    iters = st.number_input("Max solver evaluations", min_value=200, value=50000, step=1000)
    gtol = st.number_input("Gradient tolerance (smaller = more precise)", min_value=1e-20, value=1e-12, format="%.2e", step=1e-13)
    show_mats = st.checkbox("Show full P and Q matrices", value=False)
    verbose = st.checkbox("Show solver progress", value=False)
    solve_btn = st.button("Solve: dC/dy_i = 0")

# --- Build two-value P matrix ---
N = int(n*(n-1))       # ordered pairs
M = int(n*(n-1)//2)    # unordered pairs
t = 1.0 / N

rng = np.random.default_rng(42)  # Fixed seed for reproducibility

if config_type == "Random pairs":
    # Random assignment of β fraction to distance a
    pairs = [(i,j) for i in range(n) for j in range(i+1, n)]
    idx = np.arange(M); rng.shuffle(idx)
    m_a = int(round(beta * M))
    a_idx = set(idx[:m_a].tolist())
    is_a = np.zeros((n,n), dtype=bool)
    for k, (i,j) in enumerate(pairs):
        if k in a_idx:
            is_a[i,j] = True; is_a[j,i] = True

    rho = exp(-(a*a - 1.0)/(2.0*sigma*sigma))
    den = (1.0 - beta) + beta * rho
    p1 = 1.0 / (N * den)
    pa = rho / (N * den)

    P = np.where(is_a, pa, p1).astype(float)
    # Add small epsilon to avoid exactly zero probabilities
    P = np.maximum(P, 1e-300)

else:  # Star configuration
    # n_close points at distance 1, n_far point(s) at distance d_star
    far_indices = list(range(n - int(n_far), n))  # Last n_far points are "far"
    close_indices = list(range(n - int(n_far)))   # First n_close points are "close"

    is_star = np.zeros((n,n), dtype=bool)

    # Far points to close points at distance d_star
    for i in far_indices:
        for j in close_indices:
            is_star[i,j] = True
            is_star[j,i] = True

    # Far points to each other at distance d_star
    for i in far_indices:
        for j in far_indices:
            if i != j:
                is_star[i,j] = True

    # All close points to each other at distance 1 (is_star remains False)

    rho_star = exp(-(d_star*d_star - 1.0)/(2.0*sigma*sigma))
    # Count pairs at distance d_star: close<->far + far<->far
    n_close_far_pairs = 2 * int(n_close) * int(n_far)  # ordered pairs close<->far
    n_far_far_pairs = int(n_far) * (int(n_far) - 1)    # ordered pairs far<->far
    n_star_pairs = n_close_far_pairs + n_far_far_pairs
    n_unit_pairs = N - n_star_pairs  # remaining pairs (close<->close) at distance 1

    den = n_unit_pairs + n_star_pairs * rho_star
    p1 = 1.0 / den  # distance 1 probability
    p_star = rho_star / den  # distance d_star probability

    P = np.where(is_star, p_star, p1).astype(float)
    # Add small epsilon to avoid exactly zero probabilities
    P = np.maximum(P, 1e-300)

np.fill_diagonal(P, 0.0)

st.subheader("High-dimensional P (two values, ordered pairs)")
st.markdown("**Note:** We define P directly from pairwise distances (no explicit x_i needed). P represents the target similarity structure.")

if config_type == "Random pairs":
    st.write(f"**Selected pairs at distance {a}:**")
    selected_pairs = [(i,j) for k, (i,j) in enumerate(pairs) if k in a_idx]
    if len(selected_pairs) <= 20:
        st.write(selected_pairs)
    else:
        st.write(f"Showing first 20 of {len(selected_pairs)} pairs: {selected_pairs[:20]}")
    st.write("")
    st.write(pd.DataFrame([
        {"pair type": "distance 1", "p_ij": p1, "count (ordered)": int((1.0-beta)*N)},
        {"pair type": f"distance {a}", "p_ij": pa, "count (ordered)": int(beta*N)},
        {"pair type": "total", "sum p_ij": (1.0-beta)*N*p1 + beta*N*pa, "count (ordered)": N},
    ]))
else:  # Star configuration
    st.write(f"**Selected indices:**")
    st.write(f"- Close points (distance 1 among themselves): {close_indices}")
    st.write(f"- Far points (distance {d_star} to close & to each other): {far_indices}")
    st.write("")
    st.write(pd.DataFrame([
        {"pair type": "distance 1 (close-close)", "p_ij": p1, "count (ordered)": n_unit_pairs},
        {"pair type": f"distance {d_star} (close-far)", "p_ij": p_star, "count (ordered)": n_close_far_pairs},
        {"pair type": f"distance {d_star} (far-far)", "p_ij": p_star, "count (ordered)": n_far_far_pairs},
        {"pair type": "total", "sum p_ij": n_unit_pairs*p1 + n_star_pairs*p_star, "count (ordered)": N},
    ]))

if show_mats:
    st.write("### Matrix P (high-dimensional probabilities p_ij)")
    st.dataframe(pd.DataFrame(P))

# --- Initialization: n DISTINCT points (regular n-gon + fixed jitter) ---
def init_ngon(n, radius=1.0, jitter=1e-3):
    angles = np.linspace(0.0, 2.0*pi, n, endpoint=False)
    Y = np.stack([radius*np.cos(angles), radius*np.sin(angles)], axis=1)
    # Add fixed jitter pattern (deterministic)
    jitter_pattern = np.array([[np.cos(2*pi*i/n + 0.5), np.sin(2*pi*i/n + 0.5)] for i in range(n)])
    Y += jitter * jitter_pattern
    return Y

# --- Definitions of q, gradient, KL ---
def compute_W_Q(Y):
    # center to remove translation redundancy
    Y = Y - np.mean(Y, axis=0, keepdims=True)
    D2 = np.sum((Y[:,None,:] - Y[None,:,:])**2, axis=2)
    np.fill_diagonal(D2, 0.0)
    W = 1.0 / (1.0 + D2)
    np.fill_diagonal(W, 0.0)
    Z = np.sum(W)
    Q = W / Z if Z > 0 else np.zeros_like(W)
    return W, Q

def grad_flat(Y):
    W, Q = compute_W_Q(Y)
    G = np.zeros_like(Y)
    for i in range(n):
        diff = Y[i] - Y            # n×2
        coeff = (P[i] - Q[i]) * W[i]  # n
        G[i] = 4.0 * np.sum(coeff[:,None] * diff, axis=0)
    return G.reshape(-1)

def kl_terms(Y):
    _, Q = compute_W_Q(Y)
    eps = 1e-300
    mask = ~np.eye(n, dtype=bool)
    vecP = P[mask]
    vecQ = Q[mask] + eps

    # Handle p_ij = 0 case: 0 * log(0) = 0 by convention
    # Only compute where p_ij > 0 to avoid numerical issues
    result = np.zeros_like(vecP)
    nonzero_mask = vecP > eps
    result[nonzero_mask] = vecP[nonzero_mask] * np.log(vecP[nonzero_mask] / vecQ[nonzero_mask])
    return result

def residuals(y_flat):
    Y = y_flat.reshape(n,2)
    return grad_flat(Y)  # solve for dC/dy_i = 0

solution = None

if solve_btn:
    if not SCIPY_OK:
        st.error("SciPy is required. Please ensure scipy is available in the environment.")
    else:
        status_text = st.empty()
        progress_placeholder = st.empty() if verbose else None
        status_text.text("Solving: dC/dy_i = 0...")

        Y0 = init_ngon(n, radius=1.0, jitter=1e-2)

        res = least_squares(
            fun=residuals,
            x0=Y0.reshape(-1),
            method="trf",
            xtol=1e-20,      # Very tight position tolerance
            ftol=1e-20,      # Very tight function tolerance
            gtol=gtol,       # User-specified gradient tolerance (key for dC/dy_i = 0)
            max_nfev=int(iters),
            verbose=2 if verbose else 0,
            loss='linear',   # Use linear loss for better convergence
            tr_solver='lsmr' # Use LSMR for large problems
        )

        Yhat = res.x.reshape(n,2)

        status_text.text("Solution found!")
        solution = {"Y": Yhat, "info": res, "Y0": Y0}

if solution is not None and solution["Y"] is not None:
    Y = solution["Y"]
    W, Q = compute_W_Q(Y)
    gvec = grad_flat(Y)
    Cval = float(np.sum(kl_terms(Y)))

    st.subheader("Solution diagnostics")
    st.write("**Gradient formula:** dC/dy_i = 4∑_{j≠i}(p_ij-q_ij)(y_i-y_j)/(1+||y_i-y_j||²)")
    st.write("Solver finds Y where **dC/dy_i = 0** for all i. C(Y) is shown for reference (not constrained to 0).")
    st.write("")

    # Compute per-point gradient norms to show individual convergence
    gvec_2d = gvec.reshape(n, 2)
    per_point_grad_norms = np.linalg.norm(gvec_2d, axis=1)

    st.write(pd.DataFrame([{
        "KL cost C(Y)": Cval,
        "||dC/dy||₂": float(norm(gvec)),
        "max |dC/dy_i|": float(np.max(np.abs(gvec))),
        "max ||dC/dy_i||₂ (per point)": float(np.max(per_point_grad_norms)),
        "function evals used": solution["info"].nfev,
        "max function evals": int(iters),
        "solver status": solution["info"].status,
        "message": solution["info"].message
    }]))

    # Show warning if solver didn't converge well
    if float(norm(gvec)) > 1e-6:
        st.warning(f"⚠️ Gradient norm {float(norm(gvec)):.2e} is high. Consider increasing max solver evaluations or adjusting gradient tolerance.")
    elif float(norm(gvec)) > gtol * 10:
        st.info(f"ℹ️ Gradient norm {float(norm(gvec)):.2e} is above target tolerance {gtol:.2e}. Solver may have stopped early.")
    else:
        st.success(f"✓ Achieved gradient norm {float(norm(gvec)):.2e} (target: {gtol:.2e})")

    st.subheader("Low-dimensional embedding: pairwise distances")

    # Compute pairwise distances
    D = np.sqrt(np.sum((Y[:,None,:] - Y[None,:,:])**2, axis=2))
    D_offdiag = D[~np.eye(n, dtype=bool)]

    # Count unique distances
    unique_distances = np.unique(np.round(D_offdiag, 12))  # Round to avoid floating point precision issues
    num_unique_distances = len(unique_distances)

    distance_stats = pd.DataFrame([{
        "metric": "Pairwise distances ||y_i - y_j||",
        "unique distance values": num_unique_distances,
        "total pairs": len(D_offdiag),
        "min distance": float(np.min(D_offdiag)),
        "max distance": float(np.max(D_offdiag)),
        "mean distance": float(np.mean(D_offdiag))
    }])
    st.write(distance_stats)

    st.write(f"**Number of different distance values: {num_unique_distances}** (out of {len(D_offdiag)} total pairs)")

    # Show all pairwise distances
    st.write("**All pairwise distances ||y_i - y_j||:**")

    # Create a dataframe with all pairwise distances
    distance_data = []
    for i in range(n):
        for j in range(i+1, n):  # Only upper triangle to avoid duplicates
            distance_data.append({
                "i": i,
                "j": j,
                "||y_i - y_j||": D[i, j]
            })

    distance_df = pd.DataFrame(distance_data)
    st.dataframe(distance_df, height=400)

    # Also show Q statistics for reference
    st.write("")
    st.write("**Q matrix statistics (for reference):**")
    Q_offdiag = Q[~np.eye(n, dtype=bool)]
    st.write(f"- Formula: q_ij = w_ij / Z where w_ij = 1/(1 + ||y_i - y_j||²) and Z = Σw_ij")
    st.write(f"- min q_ij: {float(np.min(Q_offdiag)):.10f}")
    st.write(f"- max q_ij: {float(np.max(Q_offdiag)):.10f}")
    st.write(f"- mean q_ij: {float(np.mean(Q_offdiag)):.10f}")
    st.write(f"- sum q_ij: {float(np.sum(Q_offdiag)):.10f}")

    if show_mats:
        st.write("### Matrix Q (low-dimensional probabilities q_ij from solved Y)")
        st.dataframe(pd.DataFrame(Q))

    # Show initial Y_0 coordinates
    st.subheader("Initial coordinates Y₀ (before optimization)")
    Y0 = solution.get("Y0", None)
    if Y0 is not None:
        Y0_df = pd.DataFrame(Y0, columns=["y₁", "y₂"])
        Y0_df.insert(0, "point_index", range(n))
        if config_type == "Star configuration":
            Y0_df["group"] = ["close" if i in close_indices else "far" for i in range(n)]
            st.write("Initial Y₀ (regular n-gon with fixed jitter):")
        st.dataframe(Y0_df)

    # Visualization
    st.subheader("Solved embedding Y (final coordinates)")

    fig = go.Figure()

    if config_type == "Star configuration":
        # Close points
        close_y = Y[close_indices]
        fig.add_trace(go.Scatter(
            x=close_y[:, 0], y=close_y[:, 1],
            mode='markers+text',
            marker=dict(size=12, color='blue'),
            text=[str(i) for i in close_indices],
            textposition="top center",
            name='close',
            showlegend=True
        ))
        # Far points
        far_y = Y[far_indices]
        fig.add_trace(go.Scatter(
            x=far_y[:, 0], y=far_y[:, 1],
            mode='markers+text',
            marker=dict(size=12, color='red'),
            text=[str(i) for i in far_indices],
            textposition="top center",
            name='far',
            showlegend=True
        ))
    else:
        fig.add_trace(go.Scatter(
            x=Y[:, 0], y=Y[:, 1],
            mode='markers+text',
            marker=dict(size=12, color='blue'),
            text=[str(i) for i in range(n)],
            textposition="top center",
            showlegend=False
        ))

    fig.update_layout(
        xaxis_title="y₁",
        yaxis_title="y₂",
        width=700, height=700,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Final solved Y coordinates (y_i after optimization)")
    Y_df = pd.DataFrame(Y, columns=["y₁", "y₂"])
    Y_df.insert(0, "point_index i", range(n))

    if config_type == "Star configuration":
        Y_df["group"] = ["close" if i in close_indices else "far" for i in range(n)]
        st.write("Points grouped by distance configuration (where dC/dy_i = 0):")
    else:
        st.write("Final embedding coordinates where dC/dy_i = 0:")

    st.dataframe(Y_df)

    st.subheader("Download results")
    def df_to_bytes(df): return df.to_csv(index=False).encode("utf-8")
    st.download_button("Download P.csv", data=df_to_bytes(pd.DataFrame(P)), file_name="P.csv")
    st.download_button("Download Q.csv", data=df_to_bytes(pd.DataFrame(Q)), file_name="Q.csv")
    st.download_button("Download Y.csv", data=df_to_bytes(Y_df), file_name="Y.csv")
else:
    st.info("Click solve. Initialization uses a regular n-gon with fixed jitter so all y_i are **distinct**. The solver finds Y satisfying dC/dy_i=0 for all i (critical point of KL divergence).")