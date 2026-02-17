import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from models import RegressionModel, NeuralNetworkModel, RBFModel
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="TPU Permeability Model", layout="wide")

# ============== DATA ==============
# All experimental data from Franz Cell experiments
# Phenol data: 6 membranes
PHENOL_DATA = [
    {"id": "M-01", "Sparsa1": 100, "Sparsa2":   0, "Carbosil1":  0, "Carbosil2": 0, "permeability": 1.60618e-6,  "thickness": 0.0254},
    {"id": "M-02", "Sparsa1":   0, "Sparsa2": 100, "Carbosil1":  0, "Carbosil2": 0, "permeability": 7.55954e-7,  "thickness": 0.037},
    {"id": "M-03", "Sparsa1":   0, "Sparsa2":   0, "Carbosil1":100, "Carbosil2": 0, "permeability": 1.68063e-7,  "thickness": 0.021},
    {"id": "M-05", "Sparsa1":  60, "Sparsa2":  40, "Carbosil1":  0, "Carbosil2": 0, "permeability": 5.75051e-7,  "thickness": 0.0202},
    {"id": "M-07", "Sparsa1":  30, "Sparsa2":  70, "Carbosil1":  0, "Carbosil2": 0, "permeability": 3.39749e-8,  "thickness": 0.0208},
    {"id": "M-11", "Sparsa1":  10, "Sparsa2":  20, "Carbosil1": 70, "Carbosil2": 0, "permeability": 1.59367e-7,  "thickness": 0.016},
]

# M-Cresol data: 5 membranes
MCRESOL_DATA = [
    {"id": "M-02", "Sparsa1":  0, "Sparsa2": 100, "Carbosil1":  0, "Carbosil2": 0, "permeability": 1.0215e-7,  "thickness": 0.018},
    {"id": "M-03", "Sparsa1":  0, "Sparsa2":   0, "Carbosil1":100, "Carbosil2": 0, "permeability": 7.64893e-8, "thickness": 0.0152},
    {"id": "M-07", "Sparsa1": 30, "Sparsa2":  70, "Carbosil1":  0, "Carbosil2": 0, "permeability": 9.7528e-8,  "thickness": 0.0208},
    {"id": "M-11", "Sparsa1": 10, "Sparsa2":  20, "Carbosil1": 70, "Carbosil2": 0, "permeability": 1.09746e-7, "thickness": 0.016},
    {"id": "M-15", "Sparsa1":  0, "Sparsa2":  50, "Carbosil1": 50, "Carbosil2": 0, "permeability": 1.81053e-7, "thickness": 0.0194},
]

def get_data(permeant):
    data = PHENOL_DATA if permeant == "Phenol" else MCRESOL_DATA
    df = pd.DataFrame(data)
    return df

def get_features(df):
    X = df[["Sparsa1", "Sparsa2", "Carbosil1", "Carbosil2"]].values / 100.0
    y = np.log10(df["permeability"].values)
    return X, y

# ============== MODELS ==============

@st.cache_data
def train_models(permeant):
    df = get_data(permeant)
    X, y = get_features(df)

    reg = RegressionModel().fit(X, y)
    nn  = NeuralNetworkModel().fit(X, y)
    rbf = RBFModel().fit(X, y)

    return reg, nn, rbf, X, y, df


def predict_with_model(model_name, permeant, s1, s2, c1, c2):
    reg, nn, rbf, X, y, df = train_models(permeant)
    total = s1 + s2 + c1 + c2
    if total == 0:
        return None
    x = np.array([[s1/total, s2/total, c1/total, c2/total]])
    if model_name == "Regression":
        log_p = reg.predict(x)[0]
    elif model_name == "Neural Network":
        log_p = nn.predict(x)[0]
    else:
        log_p = rbf.predict(x)[0]
    return 10 ** log_p


def find_optimal(model_name, permeant, maximize=True):
    reg, nn, rbf, X, y, df = train_models(permeant)

    def objective(x):
        x = np.array([x])
        if model_name == "Regression":
            val = reg.predict(x)[0]
        elif model_name == "Neural Network":
            val = nn.predict(x)[0]
        else:
            val = rbf.predict(x)[0]
        return -val if maximize else val

    # Constraints: sum = 1, all >= 0
    constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    bounds = [(0, 1)] * 4

    # Try multiple starts
    best_result = None
    best_val = np.inf if not maximize else -np.inf
    for _ in range(50):
        x0 = np.random.dirichlet(np.ones(4))
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 1000})
        if res.success:
            val = -res.fun if maximize else res.fun
            if (maximize and val > best_val) or (not maximize and val < best_val):
                best_val = val
                best_result = res

    if best_result is None:
        return None

    x_opt = best_result.x
    x_opt = np.maximum(x_opt, 0)
    x_opt /= x_opt.sum()
    log_p_opt = -best_result.fun if maximize else best_result.fun
    return {
        "Sparsa1":   round(x_opt[0] * 100, 1),
        "Sparsa2":   round(x_opt[1] * 100, 1),
        "Carbosil1": round(x_opt[2] * 100, 1),
        "Carbosil2": round(x_opt[3] * 100, 1),
        "permeability": 10 ** log_p_opt,
        "log_permeability": log_p_opt,
    }


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
tab_tpu, tab_perm, tab_opt = st.tabs(["TPU Membranes", "Permeability", "Optimal Composition"])


# ============== TAB 1: TPU MEMBRANES ==============
with tab_tpu:
    st.title("TPU Membrane Data")
    st.markdown("Experimental Franz Cell permeability data for TPU membrane formulations.")

    permeant_view = st.radio("Permeant", ["Phenol", "M-Cresol"], horizontal=True, key="tpu_view")
    df_view = get_data(permeant_view)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Membrane Compositions")
        disp = df_view[["id","Sparsa1","Sparsa2","Carbosil1","Carbosil2","permeability","thickness"]].copy()
        disp.columns = ["ID","Sparsa 1 (%)","Sparsa 2 (%)","Carbosil 1 (%)","Carbosil 2 (%)","Permeability (cm/s)","Thickness (cm)"]
        disp["Permeability (cm/s)"] = disp["Permeability (cm/s)"].apply(lambda x: f"{x:.3e}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.subheader("Model Performance")
        reg, nn, rbf, X, y, df = train_models(permeant_view)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Regression R²",  f"{reg.r2(X,y):.3f}")
        mc2.metric("Neural Net R²",  f"{nn.r2(X,y):.3f}")
        mc3.metric("RBF R²",         f"{rbf.r2(X,y):.3f}")

    with col2:
        st.subheader("Permeability by Membrane")
        import json
        labels = df_view["id"].tolist()
        values = [float(p) for p in df_view["permeability"].tolist()]
        bar_html = f"""
<div style="width:100%;background:#1a1a1a;border-radius:8px;padding:20px;box-sizing:border-box;">
<canvas id="barChart" width="600" height="350"></canvas>
</div>
<script>
(function(){{
    var canvas = document.getElementById('barChart');
    canvas.width = canvas.parentElement.offsetWidth - 40;
    var ctx = canvas.getContext('2d');
    var labels = {json.dumps(labels)};
    var values = {json.dumps(values)};
    var maxV = Math.max(...values);
    var pad = {{top:30, right:20, bottom:60, left:80}};
    var w = canvas.width - pad.left - pad.right;
    var h = canvas.height - pad.top - pad.bottom;
    var barW = w / labels.length * 0.6;
    var gap   = w / labels.length;
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    for(var i=0;i<=5;i++){{
        var yy = pad.top + h - (i/5)*h;
        ctx.beginPath(); ctx.moveTo(pad.left,yy); ctx.lineTo(pad.left+w,yy); ctx.stroke();
        var label = (maxV * i/5).toExponential(1);
        ctx.fillStyle='rgba(255,255,255,0.5)';
        ctx.font='11px Arial';
        ctx.textAlign='right';
        ctx.fillText(label, pad.left-6, yy+4);
    }}
    // Bars
    var colors = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6','#1abc9c'];
    for(var i=0;i<labels.length;i++){{
        var x = pad.left + i*gap + (gap-barW)/2;
        var bh = (values[i]/maxV) * h;
        var y = pad.top + h - bh;
        ctx.fillStyle = colors[i % colors.length];
        ctx.fillRect(x, y, barW, bh);
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(labels[i], x+barW/2, pad.top+h+18);
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.font = '10px Arial';
        ctx.fillText(values[i].toExponential(1), x+barW/2, y-5);
    }}
    // Axis label
    ctx.save();
    ctx.translate(14, pad.top + h/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillStyle='rgba(255,255,255,0.5)';
    ctx.font='12px Arial';
    ctx.textAlign='center';
    ctx.fillText('Permeability (cm/s)', 0, 0);
    ctx.restore();
}})();
</script>
"""
        components.html(bar_html, height=400)


# ============== TAB 2: PERMEABILITY ==============
with tab_perm:
    st.title("Permeability Calculator")
    st.markdown("Predict permeability for a given TPU membrane composition using trained models.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")
        permeant = st.selectbox("Permeant", ["Phenol", "M-Cresol"], key="perm_permeant")
        model_name = st.selectbox("Model", ["Regression", "Neural Network", "RBF Interpolation"], key="perm_model")

        st.subheader("Membrane Composition (%)")
        st.caption("Values are relative weights - they will be normalized to 100%.")
        s1 = st.slider("Sparsa 1",   0, 100, 50, key="perm_s1")
        s2 = st.slider("Sparsa 2",   0, 100,  0, key="perm_s2")
        c1 = st.slider("Carbosil 1", 0, 100, 50, key="perm_c1")
        c2 = st.slider("Carbosil 2", 0, 100,  0, key="perm_c2")

        total = s1 + s2 + c1 + c2
        if total > 0:
            st.caption(f"Normalized: Sparsa 1={s1/total*100:.1f}%  Sparsa 2={s2/total*100:.1f}%  Carbosil 1={c1/total*100:.1f}%  Carbosil 2={c2/total*100:.1f}%")
        else:
            st.warning("Set at least one component above 0.")

        if st.button("Calculate Permeability", type="primary", use_container_width=True, key="calc_perm"):
            if total == 0:
                st.error("All components are 0.")
            else:
                with st.spinner("Calculating..."):
                    p = predict_with_model(model_name, permeant, s1, s2, c1, c2)
                    st.session_state.perm_calc_result = {
                        "permeability": p,
                        "model": model_name,
                        "permeant": permeant,
                        "s1": s1, "s2": s2, "c1": c1, "c2": c2
                    }

    with col2:
        st.subheader("Results")
        if "perm_calc_result" in st.session_state and st.session_state.perm_calc_result:
            res = st.session_state.perm_calc_result
            p = res["permeability"]
            log_p = np.log10(p)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Permeability (cm/s)", f"{p:.3e}")
            mc2.metric("log P", f"{log_p:.2f}")
            if log_p > -6.5:
                mc3.metric("Classification", "High")
            elif log_p > -7.5:
                mc3.metric("Classification", "Moderate")
            else:
                mc3.metric("Classification", "Low")

            # Show which model
            st.caption(f"Model: {res['model']}  |  Permeant: {res['permeant']}")

            st.divider()
            st.subheader("Permeation Visualization")

            mol_colors = {"Phenol": "#e74c3c", "M-Cresol": "#9b59b6"}
            color = mol_colors.get(res["permeant"], "#3498db")
            render_animation(p, res["permeant"], color)

            # Compare all three models
            st.divider()
            st.subheader("Model Comparison")
            total2 = res["s1"] + res["s2"] + res["c1"] + res["c2"]
            p_reg = predict_with_model("Regression",      res["permeant"], res["s1"], res["s2"], res["c1"], res["c2"])
            p_nn  = predict_with_model("Neural Network",  res["permeant"], res["s1"], res["s2"], res["c1"], res["c2"])
            p_rbf = predict_with_model("RBF Interpolation", res["permeant"], res["s1"], res["s2"], res["c1"], res["c2"])
            cmp_df = pd.DataFrame({
                "Model": ["Regression", "Neural Network", "RBF Interpolation"],
                "Permeability (cm/s)": [f"{p_reg:.3e}", f"{p_nn:.3e}", f"{p_rbf:.3e}"],
                "log P": [f"{np.log10(p_reg):.2f}", f"{np.log10(p_nn):.2f}", f"{np.log10(p_rbf):.2f}"]
            })
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Set a composition and click Calculate Permeability.")

            # Show training data for reference
            st.subheader("Training Data")
            permeant_ref = st.session_state.get("perm_permeant", "Phenol")
            df_ref = get_data(permeant_ref)
            disp_ref = df_ref[["id","Sparsa1","Sparsa2","Carbosil1","Carbosil2","permeability"]].copy()
            disp_ref.columns = ["ID","Sparsa 1 (%)","Sparsa 2 (%)","Carbosil 1 (%)","Carbosil 2 (%)","Permeability (cm/s)"]
            disp_ref["Permeability (cm/s)"] = disp_ref["Permeability (cm/s)"].apply(lambda x: f"{x:.3e}")
            st.dataframe(disp_ref, use_container_width=True, hide_index=True)


# ============== TAB 3: OPTIMAL COMPOSITION ==============
with tab_opt:
    st.title("Optimal Composition Finder")
    st.markdown("Find the membrane composition that maximizes or minimizes permeability for a given permeant.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")
        opt_permeant    = st.selectbox("Permeant", ["Phenol", "M-Cresol"], key="opt_permeant")
        opt_model       = st.selectbox("Model", ["Regression", "Neural Network", "RBF Interpolation"], key="opt_model")
        opt_direction   = st.radio("Optimize for", ["Maximize Permeability", "Minimize Permeability"], key="opt_dir")
        maximize        = opt_direction == "Maximize Permeability"

        st.divider()
        st.subheader("Dataset")
        df_opt = get_data(opt_permeant)
        disp_opt = df_opt[["id","Sparsa1","Sparsa2","Carbosil1","Carbosil2","permeability"]].copy()
        disp_opt.columns = ["ID","Sparsa 1","Sparsa 2","Carbosil 1","Carbosil 2","Permeability (cm/s)"]
        disp_opt["Permeability (cm/s)"] = disp_opt["Permeability (cm/s)"].apply(lambda x: f"{x:.3e}")
        st.dataframe(disp_opt, use_container_width=True, hide_index=True)

        if st.button("Generate Optimal Composition", type="primary", use_container_width=True, key="gen_opt"):
            with st.spinner("Optimizing..."):
                result = find_optimal(opt_model, opt_permeant, maximize=maximize)
                st.session_state.opt_result = result
                st.session_state.opt_settings = {
                    "model": opt_model,
                    "permeant": opt_permeant,
                    "maximize": maximize
                }

    with col2:
        st.subheader("Optimal Composition")

        if "opt_result" in st.session_state and st.session_state.opt_result:
            res = st.session_state.opt_result
            cfg = st.session_state.opt_settings
            p = res["permeability"]
            log_p = res["log_permeability"]

            direction_label = "Maximum" if cfg["maximize"] else "Minimum"
            st.markdown(f"**{direction_label} permeability for {cfg['permeant']} using {cfg['model']}**")

            # Composition
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Sparsa 1",   f"{res['Sparsa1']:.1f}%")
            mc2.metric("Sparsa 2",   f"{res['Sparsa2']:.1f}%")
            mc3.metric("Carbosil 1", f"{res['Carbosil1']:.1f}%")
            mc4.metric("Carbosil 2", f"{res['Carbosil2']:.1f}%")

            st.divider()

            rc1, rc2 = st.columns(2)
            rc1.metric("Predicted Permeability (cm/s)", f"{p:.3e}")
            rc2.metric("log P", f"{log_p:.2f}")

            # Composition pie chart
            labels_pie = ["Sparsa 1", "Sparsa 2", "Carbosil 1", "Carbosil 2"]
            values_pie = [res["Sparsa1"], res["Sparsa2"], res["Carbosil1"], res["Carbosil2"]]
            colors_pie = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

            import json
            pie_html = f"""
<div style="width:100%;background:#1a1a1a;border-radius:8px;padding:20px;text-align:center;box-sizing:border-box;">
<canvas id="pieChart" width="300" height="300" style="margin:auto;display:block;"></canvas>
</div>
<script>
(function(){{
    var canvas = document.getElementById('pieChart');
    var ctx = canvas.getContext('2d');
    var values = {json.dumps(values_pie)};
    var labels = {json.dumps(labels_pie)};
    var colors = {json.dumps(colors_pie)};
    var total = values.reduce((a,b)=>a+b, 0);
    var cx = canvas.width/2, cy = canvas.height/2, r = 100;
    var start = -Math.PI/2;
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    for(var i=0; i<values.length; i++){{
        if(values[i] === 0) continue;
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
        // Label
        var mid = start + slice/2;
        var lx = cx + (r*0.65) * Math.cos(mid);
        var ly = cy + (r*0.65) * Math.sin(mid);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if(values[i] > 2) ctx.fillText(values[i].toFixed(1)+'%', lx, ly);
        start += slice;
    }}
    // Legend
    var ly2 = canvas.height - 50;
    var lx2 = 10;
    for(var i=0; i<labels.length; i++){{
        ctx.fillStyle = colors[i];
        ctx.fillRect(lx2, ly2, 12, 12);
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.font = '11px Arial';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(labels[i], lx2+16, ly2);
        lx2 += 80;
    }}
}})();
</script>
"""
            components.html(pie_html, height=340)

            # Animation
            st.divider()
            st.subheader("Permeation Visualization")
            mol_colors2 = {"Phenol": "#e74c3c", "M-Cresol": "#9b59b6"}
            color2 = mol_colors2.get(cfg["permeant"], "#3498db")
            render_animation(p, cfg["permeant"], color2)

        else:
            st.info("Click 'Generate Optimal Composition' to find the best formulation.")

            # Show all data for reference while waiting
            st.subheader("All Experimental Data")
            for perm_name in ["Phenol", "M-Cresol"]:
                st.markdown(f"**{perm_name}**")
                df_show = get_data(perm_name)
                d = df_show[["id","Sparsa1","Sparsa2","Carbosil1","Carbosil2","permeability"]].copy()
                d.columns = ["ID","Sparsa 1 (%)","Sparsa 2 (%)","Carbosil 1 (%)","Carbosil 2 (%)","Permeability (cm/s)"]
                d["Permeability (cm/s)"] = d["Permeability (cm/s)"].apply(lambda x: f"{x:.3e}")
                st.dataframe(d, use_container_width=True, hide_index=True)
