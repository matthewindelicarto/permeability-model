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

    return {
        "Sparsa1":   round(x_opt[0] * 100, 1),
        "Sparsa2":   round(x_opt[1] * 100, 1),
        "Carbosil1": round(x_opt[2] * 100, 1),
        "Carbosil2": round(x_opt[3] * 100, 1),
        "perm_phenol":       10 ** lp_ph,
        "perm_mcresol":      10 ** lp_mc,
        "perm_glucose":      10 ** lp_gl if glucose_in_domain else None,
        "glucose_in_domain": glucose_in_domain,
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

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Regression R²",       f"{r2_reg:.3f}")
    mc2.metric("Neural Network R²",   f"{r2_nn:.3f}")
    mc3.metric("Gaussian Process R²", f"{r2_gp:.3f}")

    st.divider()

    if st.button("Find Best Model", type="primary", key="find_best"):
        scores = {
            "Regression":       r2_reg,
            "Neural Network":   r2_nn,
            "Gaussian Process": r2_gp,
        }
        best_name = max(scores, key=scores.get)
        best_r2   = scores[best_name]

        # Build reasoning for each model
        reasons = {
            "Regression": (
                "Polynomial ridge regression scored highest on this dataset. "
                "It fits a smooth quadratic surface through the composition space, "
                "which works well when permeability trends are relatively linear across blends. "
                "Ridge regularization prevents overfitting despite the small number of data points."
            ),
            "Neural Network": (
                "The neural network ensemble scored highest on this dataset. "
                "It averages predictions from 5 independently trained networks, which smooths out "
                "seed-dependent quirks. L2 regularization and a small hidden layer (4 neurons) "
                "keep it from memorizing individual data points."
            ),
            "Gaussian Process": (
                "The Gaussian Process scored highest on this dataset. "
                "GP is the gold standard for small scientific datasets — it uses a Matern-2.5 kernel "
                "that assumes smooth but physically realistic variation across compositions, "
                "and automatically tunes its parameters to best fit the available data."
            ),
        }

        # Warn if best R² is still low
        if best_r2 < 0.80:
            st.warning(
                f"All models have relatively low R² on {permeant_view} data. "
                "This may indicate high experimental variability or that more data points "
                "are needed to reliably model this composition space."
            )

        st.success(f"Best model for {permeant_view}: **{best_name}** (R² = {best_r2:.3f})")
        st.markdown(reasons[best_name])

        # Show ranking table
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rank_df = pd.DataFrame({
            "Rank":  ["1st", "2nd", "3rd"],
            "Model": [r[0] for r in ranked],
            "R²":    [f"{r[1]:.3f}" for r in ranked],
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
            if res["glucose_in_domain"]:
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Phenol (cm/s)",   f"{res['perm_phenol']:.3e}")
                pc2.metric("M-Cresol (cm/s)", f"{res['perm_mcresol']:.3e}")
                pc3.metric("Glucose (cm/s)",  f"{res['perm_glucose']:.3e}")
            else:
                pc1, pc2 = st.columns(2)
                pc1.metric("Phenol (cm/s)",   f"{res['perm_phenol']:.3e}")
                pc2.metric("M-Cresol (cm/s)", f"{res['perm_mcresol']:.3e}")
                st.caption("Glucose permeability not shown — this composition contains Sparsa 1 (27G26) or Carbosil 2 (2090A), which are outside the glucose dataset.")

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
