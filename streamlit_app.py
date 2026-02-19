import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from models import RegressionModel, NeuralNetworkModel, GaussianProcessModel
import warnings
import json
warnings.filterwarnings("ignore")

st.set_page_config(page_title="TPU Permeability Model", layout="wide")

# ============== DATA ==============
PHENOL_DATA = [
    {"id": "M-01",    "Sparsa1": 100, "Sparsa2":  0, "Carbosil1":  0, "Carbosil2":  0, "permeability": 1.60618e-6,  "thickness": 0.0254},
    {"id": "M-02",    "Sparsa1":   0, "Sparsa2":100, "Carbosil1":  0, "Carbosil2":  0, "permeability": 7.55954e-7,  "thickness": 0.037},
    {"id": "M-03",    "Sparsa1":   0, "Sparsa2":  0, "Carbosil1":100, "Carbosil2":  0, "permeability": 1.68063e-7,  "thickness": 0.021},
    {"id": "M-03(3)", "Sparsa1":   0, "Sparsa2":  0, "Carbosil1":100, "Carbosil2":  0, "permeability": 2.17540e-7,  "thickness": 0.0181},
    {"id": "M-05",    "Sparsa1":  60, "Sparsa2": 40, "Carbosil1":  0, "Carbosil2":  0, "permeability": 5.75051e-7,  "thickness": 0.0202},
    {"id": "M-07",    "Sparsa1":  30, "Sparsa2": 70, "Carbosil1":  0, "Carbosil2":  0, "permeability": 3.39749e-8,  "thickness": 0.0208},
    {"id": "M-11",    "Sparsa1":  10, "Sparsa2": 20, "Carbosil1": 70, "Carbosil2":  0, "permeability": 1.59367e-7,  "thickness": 0.016},
]

MCRESOL_DATA = [
    {"id": "M-02",    "Sparsa1":  0, "Sparsa2":100, "Carbosil1":  0, "Carbosil2":  0, "permeability": 1.02150e-7,  "thickness": 0.018},
    {"id": "M-04(2)", "Sparsa1":  0, "Sparsa2":  0, "Carbosil1":  0, "Carbosil2":100, "permeability": 1.46250e-7,  "thickness": 0.0096},
    {"id": "M-07",    "Sparsa1": 30, "Sparsa2": 70, "Carbosil1":  0, "Carbosil2":  0, "permeability": 9.75280e-8,  "thickness": 0.0208},
    {"id": "M-09",    "Sparsa1": 60, "Sparsa2": 10, "Carbosil1": 30, "Carbosil2":  0, "permeability": 5.05120e-8,  "thickness": 0.0194},
    {"id": "M-11",    "Sparsa1": 10, "Sparsa2": 20, "Carbosil1": 70, "Carbosil2":  0, "permeability": 1.09746e-7,  "thickness": 0.016},
    {"id": "M-15",    "Sparsa1":  0, "Sparsa2": 50, "Carbosil1": 50, "Carbosil2":  0, "permeability": 1.81053e-7,  "thickness": 0.0194},
    {"id": "M-20",    "Sparsa1":  0, "Sparsa2": 60, "Carbosil1": 20, "Carbosil2": 20, "permeability": 1.82590e-7,  "thickness": 0.014},
    {"id": "M-22",    "Sparsa1": 37, "Sparsa2": 63, "Carbosil1":  0, "Carbosil2":  0, "permeability": 4.13480e-7,  "thickness": 0.015},
]

# Glucose data only covers Sparsa2 + Carbosil1 blends (Sparsa1=0, Carbosil2=0)
GLUCOSE_DATA = [
    {"id": "G-01", "Sparsa1": 0, "Sparsa2":   0, "Carbosil1": 100, "Carbosil2": 0, "permeability": 1.00e-13},
    {"id": "G-02", "Sparsa1": 0, "Sparsa2":  30, "Carbosil1":  70, "Carbosil2": 0, "permeability": 8.30e-11},
    {"id": "G-03", "Sparsa1": 0, "Sparsa2":  40, "Carbosil1":  60, "Carbosil2": 0, "permeability": 7.80e-10},
    {"id": "G-04", "Sparsa1": 0, "Sparsa2":  60, "Carbosil1":  40, "Carbosil2": 0, "permeability": 9.68e-9},
    {"id": "G-05", "Sparsa1": 0, "Sparsa2":  80, "Carbosil1":  20, "Carbosil2": 0, "permeability": 2.12e-8},
    {"id": "G-06", "Sparsa1": 0, "Sparsa2": 100, "Carbosil1":   0, "Carbosil2": 0, "permeability": 2.19e-8},
]

def get_data(permeant):
    if permeant == "Phenol":
        return pd.DataFrame(PHENOL_DATA)
    elif permeant == "M-Cresol":
        return pd.DataFrame(MCRESOL_DATA)
    else:
        return pd.DataFrame(GLUCOSE_DATA)

def get_features(df):
    X = df[["Sparsa1", "Sparsa2", "Carbosil1", "Carbosil2"]].values / 100.0
    y = np.log10(df["permeability"].values)
    return X, y

# ============== MODELS ==============

@st.cache_resource
def train_models(permeant):
    df = get_data(permeant)
    X, y = get_features(df)
    reg = RegressionModel().fit(X, y)
    nn  = NeuralNetworkModel().fit(X, y)
    gp  = GaussianProcessModel().fit(X, y)
    return reg, nn, gp, X, y, df


@st.cache_resource
def loo_cv_scores(permeant):
    """Leave-one-out cross-validation R² for each model.
    Trains on N-1 points, predicts the held-out point, repeats for all N.
    This is the realistic out-of-sample accuracy estimate for small datasets."""
    df = get_data(permeant)
    X, y = get_features(df)
    N = len(X)

    preds_reg = np.zeros(N)
    preds_nn  = np.zeros(N)
    preds_gp  = np.zeros(N)

    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[i:i+1]

        preds_reg[i] = RegressionModel().fit(X_tr, y_tr).predict(X_te)[0]
        preds_nn[i]  = NeuralNetworkModel().fit(X_tr, y_tr).predict(X_te)[0]
        preds_gp[i]  = GaussianProcessModel().fit(X_tr, y_tr).predict(X_te)[0]

    def safe_r2(y_true, y_pred):
        # r2_score can be arbitrarily negative for bad models — cap at -9.99 for display
        return max(r2_score(y_true, y_pred), -9.99)

    return {
        "Regression":       safe_r2(y, preds_reg),
        "Neural Network":   safe_r2(y, preds_nn),
        "Gaussian Process": safe_r2(y, preds_gp),
    }


def predict_with_model(model_name, permeant, s1, s2, c1, c2):
    reg, nn, gp, X, y, df = train_models(permeant)
    total = s1 + s2 + c1 + c2
    if total == 0:
        return None
    x = np.array([[s1/total, s2/total, c1/total, c2/total]])
    if model_name == "Regression":
        log_p = reg.predict(x)[0]
    elif model_name == "Neural Network":
        log_p = nn.predict(x)[0]
    else:
        log_p = gp.predict(x)[0]
    return 10 ** log_p


def find_optimal_combined(model_name):
    """
    Minimize permeability to Phenol and M-Cresol across all four components.
    Glucose data only covers the Sparsa2/Carbosil1 subspace so it is NOT used
    in the optimizer — it would distort results by pushing away from Sparsa1/Carbosil2.
    After finding the optimum, glucose permeability is reported only if the result
    happens to land in the glucose-valid domain (Sparsa1≈0, Carbosil2≈0).
    """
    reg_ph, nn_ph, gp_ph, X_ph, y_ph, _ = train_models("Phenol")
    reg_mc, nn_mc, gp_mc, X_mc, y_mc, _ = train_models("M-Cresol")
    reg_gl, nn_gl, gp_gl, X_gl, y_gl, _ = train_models("Glucose")

    ph_min, ph_max = y_ph.min(), y_ph.max()
    mc_min, mc_max = y_mc.min(), y_mc.max()

    def predict_ph_mc(x_vec):
        x = np.array([x_vec])
        if model_name == "Regression":
            lp_ph = reg_ph.predict(x)[0]
            lp_mc = reg_mc.predict(x)[0]
        elif model_name == "Neural Network":
            lp_ph = nn_ph.predict(x)[0]
            lp_mc = nn_mc.predict(x)[0]
        else:
            lp_ph = gp_ph.predict(x)[0]
            lp_mc = gp_mc.predict(x)[0]
        return lp_ph, lp_mc

    def predict_glucose(x_vec):
        x = np.array([x_vec])
        if model_name == "Regression":
            return reg_gl.predict(x)[0]
        elif model_name == "Neural Network":
            return nn_gl.predict(x)[0]
        else:
            return gp_gl.predict(x)[0]

    def objective(x):
        lp_ph, lp_mc = predict_ph_mc(x)
        # Normalize each term so neither molecule dominates
        n_ph = (lp_ph - ph_min) / (ph_max - ph_min + 1e-12)
        n_mc = (lp_mc - mc_min) / (mc_max - mc_min + 1e-12)
        return n_ph + n_mc  # minimize sum (lower log = lower permeability)

    constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    bounds = [(0, 1)] * 4

    # Structured starts cover the full simplex well without needing many random samples
    fixed_starts = [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [0.5, 0.5, 0, 0], [0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5],
        [0, 0.5, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5],
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.2, 0.7, 0], [0.1, 0.1, 0.7, 0.1], [0.2, 0.3, 0.5, 0],
        [0, 0.3, 0.7, 0], [0, 0.4, 0.6, 0], [0, 0.5, 0.5, 0],
        [0.3, 0.3, 0.4, 0], [0.2, 0.2, 0.6, 0], [0.1, 0.3, 0.6, 0],
    ]
    np.random.seed(0)
    n_random = 15 if model_name in ("Neural Network", "Gaussian Process") else 30
    random_starts = [np.random.dirichlet(np.ones(4)) for _ in range(n_random)]
    all_starts = [np.array(s, dtype=float) for s in fixed_starts] + random_starts

    best_result = None
    best_val = np.inf
    for x0 in all_starts:
        x0 = x0 / x0.sum()
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 500})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_result = res

    if best_result is None:
        return None

    x_opt = best_result.x
    x_opt = np.maximum(x_opt, 0)
    x_opt /= x_opt.sum()

    lp_ph, lp_mc = predict_ph_mc(x_opt)

    # Report glucose only if the result is in the glucose dataset's valid domain
    glucose_in_domain = (x_opt[0] < 0.05) and (x_opt[3] < 0.05)
    lp_gl = predict_glucose(x_opt) if glucose_in_domain else None

    # GP uncertainty at the optimal point (±1 std in log space → multiplicative factor in linear space)
    gp_uncertainty = None
    if model_name == "Gaussian Process":
        x_q = np.atleast_2d(x_opt)
        _, std_ph = gp_ph.model.predict(x_q, return_std=True)
        _, std_mc = gp_mc.model.predict(x_q, return_std=True)
        gp_uncertainty = {
            "std_log_phenol":  float(std_ph[0]),
            "std_log_mcresol": float(std_mc[0]),
            # Confidence interval in linear space: P * 10^(±2σ)
            "phenol_lo":  10 ** (lp_ph - 2 * float(std_ph[0])),
            "phenol_hi":  10 ** (lp_ph + 2 * float(std_ph[0])),
            "mcresol_lo": 10 ** (lp_mc - 2 * float(std_mc[0])),
            "mcresol_hi": 10 ** (lp_mc + 2 * float(std_mc[0])),
        }
        if glucose_in_domain:
            _, std_gl = gp_gl.model.predict(x_q, return_std=True)
            gp_uncertainty["glucose_lo"] = 10 ** (lp_gl - 2 * float(std_gl[0]))
            gp_uncertainty["glucose_hi"] = 10 ** (lp_gl + 2 * float(std_gl[0]))

    return {
        "Sparsa1":   round(x_opt[0] * 100, 1),
        "Sparsa2":   round(x_opt[1] * 100, 1),
        "Carbosil1": round(x_opt[2] * 100, 1),
        "Carbosil2": round(x_opt[3] * 100, 1),
        "perm_phenol":       10 ** lp_ph,
        "perm_mcresol":      10 ** lp_mc,
        "perm_glucose":      10 ** lp_gl if glucose_in_domain else None,
        "glucose_in_domain": glucose_in_domain,
        "gp_uncertainty":    gp_uncertainty,
    }


# ============== ANIMATION ==============

def render_animation(permeability, mol_name, color):
    log_p = np.log10(permeability) if permeability > 0 else -10
    base_speed = 2.0
    membrane_speed = max(0.05, min(2.0, (log_p + 10) / 8 * 2.0))
    html = f"""
<div id="anim_container" style="width:100%;height:300px;background:#1a1a1a;border-radius:8px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:10px;left:50%;transform:translateX(-50%);color:rgba(255,255,255,0.5);font-size:11px;font-family:Arial;">External</div>
    <div style="position:absolute;top:35%;left:0;right:0;height:30%;background:linear-gradient(180deg,rgba(100,100,150,0.3),rgba(60,60,110,0.6),rgba(100,100,150,0.3));"></div>
    <div style="position:absolute;top:48%;left:50%;transform:translate(-50%,-50%);color:rgba(255,255,255,0.4);font-size:13px;font-family:Arial;">TPU Membrane</div>
    <div style="position:absolute;bottom:10px;left:50%;transform:translateX(-50%);color:rgba(255,255,255,0.5);font-size:11px;font-family:Arial;">Internal</div>
</div>
<script>
(function(){{
    var container = document.getElementById('anim_container');
    var molecules = [];
    var baseSpeed = {base_speed};
    var membraneSpeed = {membrane_speed};
    var color = '{color}';
    for(var i=0; i<15; i++){{
        var mol = document.createElement('div');
        var size = 7 + Math.random()*5;
        mol.style.cssText = 'position:absolute;border-radius:50%;background:'+color+';box-shadow:0 0 8px '+color+';width:'+size+'px;height:'+size+'px;';
        mol.xpos = 10 + Math.random()*80;
        mol.ypos = Math.random()*300;
        mol.style.left = mol.xpos + '%';
        mol.style.top  = mol.ypos + 'px';
        container.appendChild(mol);
        molecules.push(mol);
    }}
    function animate(){{
        for(var i=0; i<molecules.length; i++){{
            var mol = molecules[i];
            var inMem = (mol.ypos > 105 && mol.ypos < 195);
            mol.ypos += inMem ? membraneSpeed : baseSpeed;
            if(mol.ypos > 310) mol.ypos = -15;
            mol.style.top = mol.ypos + 'px';
            mol.style.opacity = inMem ? '0.45' : '0.9';
            mol.style.transform = inMem ? 'scale(0.8)' : 'scale(1)';
        }}
        requestAnimationFrame(animate);
    }}
    animate();
}})();
</script>
"""
    components.html(html, height=320)


# ============== TABS ==============
tab_tpu, tab_perm, tab_opt = st.tabs(["Model Performance", "Permeability", "Optimal Composition"])


# ============== TAB 1: MODEL PERFORMANCE ==============
with tab_tpu:
    st.title("Model Performance")
    st.markdown("Compare how well each model fits the experimental permeability data.")

    permeant_view = st.radio("Molecule", ["Phenol", "M-Cresol", "Glucose"], horizontal=True, key="tpu_view")

    reg, nn, gp, X, y, df = train_models(permeant_view)

    r2_reg = reg.r2(X, y)
    r2_nn  = nn.r2(X, y)
    r2_gp  = gp.r2(X, y)

    st.subheader("Training R²")
    st.caption("How well each model fits the data it was trained on. High values here can mean overfitting — see LOO R² below for the realistic score.")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Regression",       f"{r2_reg:.3f}")
    mc2.metric("Neural Network",   f"{r2_nn:.3f}")
    mc3.metric("Gaussian Process", f"{r2_gp:.3f}")

    st.divider()

    st.subheader("Leave-One-Out (LOO) Cross-Validation R²")
    st.caption(
        "Trains on N−1 points, predicts the held-out point, repeats for every point. "
        "This is the honest out-of-sample accuracy — what to expect when predicting a membrane you haven't tested yet."
    )

    with st.spinner("Running LOO cross-validation..."):
        loo = loo_cv_scores(permeant_view)

    lc1, lc2, lc3 = st.columns(3)

    def loo_delta(train_r2, loo_r2):
        """Delta string for metric widget — shows gap vs training R²."""
        diff = loo_r2 - train_r2
        return f"{diff:+.3f} vs training"

    lc1.metric("Regression",       f"{loo['Regression']:.3f}",       loo_delta(r2_reg, loo['Regression']),       delta_color="normal")
    lc2.metric("Neural Network",   f"{loo['Neural Network']:.3f}",   loo_delta(r2_nn,  loo['Neural Network']),   delta_color="normal")
    lc3.metric("Gaussian Process", f"{loo['Gaussian Process']:.3f}", loo_delta(r2_gp,  loo['Gaussian Process']), delta_color="normal")

    # Contextual interpretation banner
    best_loo_name = max(loo, key=loo.get)
    best_loo_val  = loo[best_loo_name]
    if best_loo_val >= 0.85:
        st.success(f"Strong predictive accuracy — **{best_loo_name}** generalises well (LOO R² = {best_loo_val:.3f}).")
    elif best_loo_val >= 0.60:
        st.warning(
            f"Moderate predictive accuracy — best LOO R² is {best_loo_val:.3f} (**{best_loo_name}**). "
            "Predictions for untested compositions carry meaningful uncertainty. "
            "More data points would improve reliability."
        )
    else:
        st.error(
            f"Low predictive accuracy — best LOO R² is {best_loo_val:.3f} (**{best_loo_name}**). "
            "The models struggle to generalise beyond the training points. "
            "Treat optimal composition suggestions with caution and prioritise adding more experimental data."
        )

    st.divider()

    if st.button("Find Best Model", type="primary", key="find_best"):
        # Rank by LOO R² (the honest score), not training R²
        best_name = max(loo, key=loo.get)
        best_loo  = loo[best_name]
        best_train = {"Regression": r2_reg, "Neural Network": r2_nn, "Gaussian Process": r2_gp}[best_name]

        reasons = {
            "Regression": (
                "Polynomial ridge regression has the best LOO cross-validation score on this dataset. "
                "It fits a smooth quadratic surface through the composition space, which generalises well "
                "when permeability trends are roughly linear across blends. "
                "Ridge regularisation shrinks coefficients to prevent overfitting on small datasets."
            ),
            "Neural Network": (
                "The neural network ensemble has the best LOO cross-validation score on this dataset. "
                "Averaging 7 independently trained networks reduces seed-dependent variance, "
                "and the Adam optimiser handles the tight permeability range more robustly than SGD."
            ),
            "Gaussian Process": (
                "The Gaussian Process has the best LOO cross-validation score on this dataset. "
                "GP is the gold standard for small scientific datasets — the Matern-2.5 kernel "
                "assumes physically realistic smoothness across compositions, and the model "
                "automatically tunes all hyperparameters by maximising the marginal likelihood."
            ),
        }

        st.success(f"Best generalising model for {permeant_view}: **{best_name}**")
        col_a, col_b = st.columns(2)
        col_a.metric("Training R²", f"{best_train:.3f}")
        col_b.metric("LOO R²",      f"{best_loo:.3f}", f"{best_loo - best_train:+.3f} vs training", delta_color="normal")
        st.markdown(reasons[best_name])

        # Full ranking table (by LOO R²)
        train_scores = {"Regression": r2_reg, "Neural Network": r2_nn, "Gaussian Process": r2_gp}
        ranked = sorted(loo.items(), key=lambda x: x[1], reverse=True)
        rank_df = pd.DataFrame({
            "Rank":       ["1st", "2nd", "3rd"],
            "Model":      [r[0] for r in ranked],
            "Training R²":[f"{train_scores[r[0]]:.3f}" for r in ranked],
            "LOO R²":     [f"{r[1]:.3f}" for r in ranked],
        })
        st.dataframe(rank_df, use_container_width=True, hide_index=True)


# ============== TAB 2: PERMEABILITY ==============
with tab_perm:
    st.title("Permeability Calculator")
    st.markdown("Predict permeability for a given TPU membrane composition using trained models.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")
        permeant = st.selectbox("Molecule", ["Phenol", "M-Cresol", "Glucose"], key="perm_permeant")
        model_name = st.selectbox("Model", ["Regression", "Neural Network", "Gaussian Process"], key="perm_model")

        st.subheader("Membrane Composition")
        st.caption("Enter values for each component. Total must equal 100%.")

        # Four number inputs — all independent, validated to sum to 100
        s1 = st.number_input("Sparsa 1 - 27G26 (%)",   min_value=0, max_value=100, value=50, step=1, key="perm_s1")
        s2 = st.number_input("Sparsa 2 - 30G25 (%)",   min_value=0, max_value=100, value=0,  step=1, key="perm_s2")
        c1 = st.number_input("Carbosil 1 - 2080A (%)", min_value=0, max_value=100, value=50, step=1, key="perm_c1")
        c2 = st.number_input("Carbosil 2 - 2090A (%)", min_value=0, max_value=100, value=0,  step=1, key="perm_c2")

        total = s1 + s2 + c1 + c2
        if total == 100:
            st.success(f"Total: {total}% ✓")
        elif total < 100:
            st.warning(f"Total: {total}% — needs {100 - total}% more to reach 100%.")
        else:
            st.error(f"Total: {total}% — over by {total - 100}%. Reduce values to reach 100%.")

        calc_disabled = (total != 100)
        if st.button("Calculate Permeability", type="primary", use_container_width=True,
                     key="calc_perm", disabled=calc_disabled):
            with st.spinner("Calculating..."):
                p = predict_with_model(model_name, permeant, s1, s2, c1, c2)
                st.session_state.perm_calc_result = {
                    "permeability": p,
                    "model": model_name,
                    "permeant": permeant,
                    "s1": s1, "s2": s2, "c1": c1, "c2": c2,
                }

    with col2:
        st.subheader("Results")
        if "perm_calc_result" in st.session_state and st.session_state.perm_calc_result:
            res = st.session_state.perm_calc_result
            p = res["permeability"]

            st.metric("Permeability (cm/s)", f"{p:.3e}")
            st.caption(f"Model: {res['model']}  |  Molecule: {res['permeant']}")

            st.divider()
            st.subheader("Permeation Visualization")
            mol_colors = {"Phenol": "#e74c3c", "M-Cresol": "#9b59b6", "Glucose": "#f39c12"}
            color = mol_colors.get(res["permeant"], "#3498db")
            render_animation(p, res["permeant"], color)

            st.divider()
            st.subheader("Model Comparison")
            p_reg = predict_with_model("Regression",        res["permeant"], res["s1"], res["s2"], res["c1"], res["c2"])
            p_nn  = predict_with_model("Neural Network",    res["permeant"], res["s1"], res["s2"], res["c1"], res["c2"])
            p_gp  = predict_with_model("Gaussian Process",  res["permeant"], res["s1"], res["s2"], res["c1"], res["c2"])
            cmp_df = pd.DataFrame({
                "Model": ["Regression", "Neural Network", "Gaussian Process"],
                "Permeability (cm/s)": [f"{p_reg:.3e}", f"{p_nn:.3e}", f"{p_gp:.3e}"],
            })
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Set a composition totalling 100% and click Calculate Permeability.")


# ============== TAB 3: OPTIMAL COMPOSITION ==============
with tab_opt:
    st.title("Optimal Composition Finder")
    st.markdown(
        "Finds the single membrane composition that best minimizes permeability "
        "to **Phenol**, **M-Cresol**, and **Glucose** simultaneously — "
        "targeting minimum passage of all three molecules."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")
        opt_model = st.selectbox("Model", ["Regression", "Neural Network", "Gaussian Process"], key="opt_model")

        st.divider()
        st.caption(
            "Searches all compositions summing to 100%. "
            "Glucose data only covers Sparsa 2 (30G25) / Carbosil 1 (2080A) blends — "
            "it is only factored in where applicable."
        )

        if st.button("Find Optimal Composition", type="primary", use_container_width=True, key="gen_opt"):
            with st.spinner("Optimizing across all three molecules..."):
                result = find_optimal_combined(opt_model)
                st.session_state.opt_result = result
                st.session_state.opt_settings = {"model": opt_model}

    with col2:
        st.subheader("Optimal Composition")

        if "opt_result" in st.session_state and st.session_state.opt_result:
            res = st.session_state.opt_result
            cfg = st.session_state.opt_settings

            st.markdown(f"**Model used: {cfg['model']}**")

            # Composition metrics
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("Sparsa 1 (27G26)",   f"{res['Sparsa1']:.1f}%")
            oc2.metric("Sparsa 2 (30G25)",   f"{res['Sparsa2']:.1f}%")
            oc3.metric("Carbosil 1 (2080A)", f"{res['Carbosil1']:.1f}%")
            oc4.metric("Carbosil 2 (2090A)", f"{res['Carbosil2']:.1f}%")

            st.divider()

            # Predicted permeabilities
            st.subheader("Predicted Permeabilities")
            unc = res.get("gp_uncertainty")

            if res["glucose_in_domain"]:
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Phenol (cm/s)",   f"{res['perm_phenol']:.3e}")
                pc2.metric("M-Cresol (cm/s)", f"{res['perm_mcresol']:.3e}")
                pc3.metric("Glucose (cm/s)",  f"{res['perm_glucose']:.3e}")
                if unc:
                    pc1.caption(f"95% CI: {unc['phenol_lo']:.2e} – {unc['phenol_hi']:.2e}")
                    pc2.caption(f"95% CI: {unc['mcresol_lo']:.2e} – {unc['mcresol_hi']:.2e}")
                    if "glucose_lo" in unc:
                        pc3.caption(f"95% CI: {unc['glucose_lo']:.2e} – {unc['glucose_hi']:.2e}")
            else:
                pc1, pc2 = st.columns(2)
                pc1.metric("Phenol (cm/s)",   f"{res['perm_phenol']:.3e}")
                pc2.metric("M-Cresol (cm/s)", f"{res['perm_mcresol']:.3e}")
                if unc:
                    pc1.caption(f"95% CI: {unc['phenol_lo']:.2e} – {unc['phenol_hi']:.2e}")
                    pc2.caption(f"95% CI: {unc['mcresol_lo']:.2e} – {unc['mcresol_hi']:.2e}")
                st.caption("Glucose permeability not shown — this composition contains Sparsa 1 (27G26) or Carbosil 2 (2090A), which are outside the glucose dataset.")

            # GP uncertainty explanation box
            if unc:
                std_ph = unc["std_log_phenol"]
                std_mc = unc["std_log_mcresol"]
                severity = max(std_ph, std_mc)
                if severity < 0.3:
                    unc_level = "Low"
                    unc_color = "success"
                elif severity < 0.7:
                    unc_level = "Moderate"
                    unc_color = "warning"
                else:
                    unc_level = "High"
                    unc_color = "error"

                unc_msg = (
                    f"**GP Uncertainty at this composition — {unc_level}**  \n"
                    f"The GP predicts with ±{std_ph:.2f} log-units uncertainty for Phenol "
                    f"and ±{std_mc:.2f} log-units for M-Cresol (1σ).  \n"
                    f"This means the true permeability could be anywhere in the 95% confidence intervals shown above.  \n"
                    f"{'This composition is well-supported by nearby training data.' if severity < 0.3 else 'This composition is far from the training data — consider running an experiment here to confirm.'}"
                )
                if unc_color == "success":
                    st.success(unc_msg)
                elif unc_color == "warning":
                    st.warning(unc_msg)
                else:
                    st.error(unc_msg)

            st.divider()

            # Pie chart
            st.subheader("Membrane Composition")
            labels_pie  = ["Sparsa 1 (27G26)", "Sparsa 2 (30G25)", "Carbosil 1 (2080A)", "Carbosil 2 (2090A)"]
            values_pie  = [res["Sparsa1"], res["Sparsa2"], res["Carbosil1"], res["Carbosil2"]]
            colors_pie  = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
            pie_html = f"""
<div style="width:100%;background:#1a1a1a;border-radius:8px;padding:20px;box-sizing:border-box;display:flex;justify-content:center;">
<canvas id="optPieChart" width="340" height="320"></canvas>
</div>
<script>
(function(){{
    var canvas = document.getElementById('optPieChart');
    var ctx = canvas.getContext('2d');
    var values = {json.dumps(values_pie)};
    var labels = {json.dumps(labels_pie)};
    var colors = {json.dumps(colors_pie)};
    var total = values.reduce(function(a,b){{return a+b;}},0);
    var cx = canvas.width/2, cy = 150, r = 110;
    var start = -Math.PI/2;
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for(var i=0; i<values.length; i++){{
        if(values[i] <= 0) continue;
        var slice = (values[i]/total) * 2 * Math.PI;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, start, start+slice);
        ctx.closePath();
        ctx.fillStyle = colors[i];
        ctx.fill();
        ctx.strokeStyle = '#1a1a1a';
        ctx.lineWidth = 2;
        ctx.stroke();
        if(values[i] > 3){{
            var mid = start + slice/2;
            var lx = cx + r*0.65*Math.cos(mid);
            var ly = cy + r*0.65*Math.sin(mid);
            ctx.fillStyle = 'white';
            ctx.font = 'bold 13px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(values[i].toFixed(1)+'%', lx, ly);
        }}
        start += slice;
    }}
    // 2-row legend so all 4 labels fit
    var legRows = [[0,1],[2,3]];
    for(var row=0; row<legRows.length; row++){{
        var legY = 270 + row*22, legX = 10;
        for(var k=0; k<legRows[row].length; k++){{
            var i = legRows[row][k];
            ctx.fillStyle = colors[i];
            ctx.fillRect(legX, legY, 12, 12);
            ctx.fillStyle = 'rgba(255,255,255,0.85)';
            ctx.font = '11px Arial';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(labels[i], legX+16, legY+1);
            legX += 160;
        }}
    }}
}})();
</script>
"""
            components.html(pie_html, height=355)

        else:
            st.markdown("Click **Find Optimal Composition** to run the multi-molecule optimizer.")
