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

# Glucose data: 6 membranes (Sparsa2 + Carbosil1 only, cm^2/s)
# Stored as Sparsa1=0, Sparsa2=x, Carbosil1=y, Carbosil2=0 fractions
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
        data = PHENOL_DATA
    elif permeant == "M-Cresol":
        data = MCRESOL_DATA
    else:
        data = GLUCOSE_DATA
    return pd.DataFrame(data)

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


def find_optimal_combined(model_name):
    """
    Find the single membrane composition that minimizes the sum of normalized
    log-permeabilities for Phenol, M-Cresol, and Glucose simultaneously.
    Composition is constrained to the simplex (sum=1, all>=0).
    """
    # Pre-train all three permeant models
    reg_ph, nn_ph, rbf_ph, X_ph, y_ph, _ = train_models("Phenol")
    reg_mc, nn_mc, rbf_mc, X_mc, y_mc, _ = train_models("M-Cresol")
    reg_gl, nn_gl, rbf_gl, X_gl, y_gl, _ = train_models("Glucose")

    # Reference ranges for normalisation (use training data min/max)
    ph_min, ph_max = y_ph.min(), y_ph.max()
    mc_min, mc_max = y_mc.min(), y_mc.max()
    gl_min, gl_max = y_gl.min(), y_gl.max()

    def predict_all(x_vec):
        x = np.array([x_vec])
        if model_name == "Regression":
            lp_ph = reg_ph.predict(x)[0]
            lp_mc = reg_mc.predict(x)[0]
            lp_gl = reg_gl.predict(x)[0]
        elif model_name == "Neural Network":
            lp_ph = nn_ph.predict(x)[0]
            lp_mc = nn_mc.predict(x)[0]
            lp_gl = nn_gl.predict(x)[0]
        else:
            lp_ph = rbf_ph.predict(x)[0]
            lp_mc = rbf_mc.predict(x)[0]
            lp_gl = rbf_gl.predict(x)[0]
        return lp_ph, lp_mc, lp_gl

    def objective(x):
        lp_ph, lp_mc, lp_gl = predict_all(x)
        # Normalise each to [0,1] then sum — lower is better for all three
        n_ph = (lp_ph - ph_min) / (ph_max - ph_min + 1e-12)
        n_mc = (lp_mc - mc_min) / (mc_max - mc_min + 1e-12)
        n_gl = (lp_gl - gl_min) / (gl_max - gl_min + 1e-12)
        return n_ph + n_mc + n_gl  # minimize sum

    constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    bounds = [(0, 1)] * 4

    best_result = None
    best_val = np.inf
    np.random.seed(0)
    for _ in range(100):
        x0 = np.random.dirichlet(np.ones(4))
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"ftol": 1e-14, "maxiter": 2000})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_result = res

    if best_result is None:
        return None

    x_opt = best_result.x
    x_opt = np.maximum(x_opt, 0)
    x_opt /= x_opt.sum()

    lp_ph, lp_mc, lp_gl = predict_all(x_opt)

    return {
        "Sparsa1":   round(x_opt[0] * 100, 1),
        "Sparsa2":   round(x_opt[1] * 100, 1),
        "Carbosil1": round(x_opt[2] * 100, 1),
        "Carbosil2": round(x_opt[3] * 100, 1),
        "perm_phenol":  10 ** lp_ph,
        "perm_mcresol": 10 ** lp_mc,
        "perm_glucose": 10 ** lp_gl,
        "logp_phenol":  lp_ph,
        "logp_mcresol": lp_mc,
        "logp_glucose": lp_gl,
    }


# ============== 3D VIEWER ==============

def generate_tpu_atoms(s1_frac, s2_frac, c1_frac, c2_frac, n_chains=20, box_size=50, seed=42):
    """Generate a list of atoms for a TPU membrane with given composition fractions."""
    np.random.seed(seed)
    atoms = []
    atom_id = 1
    res_id = 1
    bx = box_size

    def add_atom(element, x, y, z, res_name):
        nonlocal atom_id
        atoms.append({
            "id": atom_id, "name": element, "element": element,
            "res_name": res_name, "res_id": res_id,
            "x": x, "y": y, "z": z
        })
        atom_id += 1
        return atom_id - 1

    def random_dir():
        theta = np.random.uniform(0, 2 * np.pi)
        phi   = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
        return (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))

    def step(x, y, z, bond_len, direction, wobble=0.3):
        dx, dy, dz = direction
        dx += np.random.uniform(-wobble, wobble)
        dy += np.random.uniform(-wobble, wobble)
        dz += np.random.uniform(-wobble, wobble)
        mag = np.sqrt(dx*dx + dy*dy + dz*dz)
        if mag > 0:
            dx, dy, dz = dx/mag*bond_len, dy/mag*bond_len, dz/mag*bond_len
        return x+dx, y+dy, z+dz, (dx/bond_len, dy/bond_len, dz/bond_len)

    for _ in range(n_chains):
        x = np.random.uniform(-bx/2 + 5, bx/2 - 5)
        y = np.random.uniform(-bx/2 + 5, bx/2 - 5)
        z = np.random.uniform(-bx/2 + 5, bx/2 - 5)
        direction = random_dir()
        r = np.random.random()
        if r < s1_frac:
            chain_type = "sparsa1"
        elif r < s1_frac + s2_frac:
            chain_type = "sparsa2"
        elif r < s1_frac + s2_frac + c1_frac:
            chain_type = "carbosil1"
        else:
            chain_type = "carbosil2"

        prev_atom = None
        for _ in range(np.random.randint(3, 6)):
            for _ in range(np.random.randint(4, 8)):
                x, y, z, direction = step(x, y, z, 1.54, direction)
                if chain_type in ("carbosil1", "carbosil2"):
                    a = add_atom("Si", x, y, z, "PDM")
                else:
                    a = add_atom("C",  x, y, z, "PEG")
                if prev_atom:
                    pass  # bonds not needed for viewer
                prev_atom = a
                x, y, z, direction = step(x, y, z, 1.43, direction)
                a = add_atom("O", x, y, z, "PEG" if chain_type.startswith("sparsa") else "PDM")
                prev_atom = a
            x, y, z, direction = step(x, y, z, 1.47, direction)
            a = add_atom("N", x, y, z, "URE")
            prev_atom = a
            x, y, z, direction = step(x, y, z, 1.33, direction)
            a = add_atom("C", x, y, z, "URE")
            prev_atom = a
            x, y, z, direction = step(x, y, z, 1.43, direction)
            a = add_atom("O", x, y, z, "URE")
            prev_atom = a
            res_id += 1
            if np.random.random() < 0.3:
                direction = random_dir()

    return atoms


def render_tpu_3dmol(s1, s2, c1, c2, height=520):
    """Render 3D viewer for a TPU membrane given integer composition percentages."""
    total = s1 + s2 + c1 + c2
    if total == 0:
        st.warning("All components are 0 — cannot render membrane.")
        return
    s1f, s2f, c1f, c2f = s1/total, s2/total, c1/total, c2/total
    carbosil_frac = c1f + c2f

    atoms = generate_tpu_atoms(s1f, s2f, c1f, c2f)

    pdb_lines = []
    for atom in atoms:
        pdb_lines.append(
            f"ATOM  {atom['id']:5d} {atom['name']:4s} {atom['res_name']:3s}  {atom['res_id']:4d}    "
            f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00          {atom['element']:>2s}"
        )
    pdb_lines.append("END")
    pdb_data = "\n".join(pdb_lines)
    pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    color_scheme = """
        viewer.setStyle({resn: 'PEG'}, {stick: {radius: 0.12, colorscheme: 'greenCarbon'}});
        viewer.setStyle({resn: 'PDM'}, {stick: {radius: 0.14, colorscheme: 'blueCarbon'}});
        viewer.setStyle({resn: 'URE'}, {stick: {radius: 0.14, colorscheme: 'orangeCarbon'}});
        viewer.setStyle({elem: 'N'},  {stick: {radius: 0.14}, sphere: {scale: 0.25, color: '0x3498db'}});
        viewer.setStyle({elem: 'SI'}, {stick: {radius: 0.16}, sphere: {scale: 0.3,  color: '0xf1c40f'}});
    """

    box_x, box_y, box_z = 40, 40, 15
    html = f"""
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <div id="viewer_opt" style="width:100%;height:{height}px;position:relative;"></div>
    <script>
        var viewer = $3Dmol.createViewer("viewer_opt", {{backgroundColor: "0x1a1a1a"}});
        var pdb = `{pdb_escaped}`;
        viewer.addModel(pdb, "pdb");
        {color_scheme}
        var hx={box_x}/2, hy={box_y}/2, hz={box_z}/2;
        var bc=0x555555, bw=1.5;
        viewer.addLine({{start:{{x:-hx,y:-hy,z:-hz}},end:{{x:hx,y:-hy,z:-hz}},color:bc,linewidth:bw}});
        viewer.addLine({{start:{{x:hx,y:-hy,z:-hz}},end:{{x:hx,y:hy,z:-hz}},color:bc,linewidth:bw}});
        viewer.addLine({{start:{{x:hx,y:hy,z:-hz}},end:{{x:-hx,y:hy,z:-hz}},color:bc,linewidth:bw}});
        viewer.addLine({{start:{{x:-hx,y:hy,z:-hz}},end:{{x:-hx,y:-hy,z:-hz}},color:bc,linewidth:bw}});
        viewer.addLine({{start:{{x:-hx,y:-hy,z:hz}},end:{{x:hx,y:-hy,z:hz}},color:bc,linewidth:bw}});
        viewer.addLine({{start:{{x:hx,y:-hy,z:hz}},end:{{x:hx,y:hy,z:hz}},color:bc,linewidth:bw}});
        viewer.addLine({{start:{{x:hx,y:hy,z:hz}},end:{{x:-hx,y:hy,z:hz}},color:bc,linewidth:bw}});
        viewer.addLine({{start:{{x:-hx,y:hy,z:hz}},end:{{x:-hx,y:-hy,z:hz}},color:bc,linewidth:bw}});
        viewer.addLine({{start:{{x:-hx,y:-hy,z:-hz}},end:{{x:-hx,y:-hy,z:hz}},color:bc,linewidth:bw}});
        viewer.addLine({{start:{{x:hx,y:-hy,z:-hz}},end:{{x:hx,y:-hy,z:hz}},color:bc,linewidth:bw}});
        viewer.addLine({{start:{{x:hx,y:hy,z:-hz}},end:{{x:hx,y:hy,z:hz}},color:bc,linewidth:bw}});
        viewer.addLine({{start:{{x:-hx,y:hy,z:-hz}},end:{{x:-hx,y:hy,z:hz}},color:bc,linewidth:bw}});
        viewer.zoomTo(); viewer.zoom(0.5);
        viewer.rotate(20,{{x:1,y:0,z:0}}); viewer.rotate(-15,{{x:0,y:1,z:0}});
        viewer.render();
    </script>
    """
    components.html(html, height=height + 20)


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

    permeant_view = st.radio("Permeant", ["Phenol", "M-Cresol", "Glucose"], horizontal=True, key="tpu_view")
    df_view = get_data(permeant_view)

    st.subheader("Model Performance")
    reg, nn, rbf, X, y, df = train_models(permeant_view)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Regression R²",  f"{reg.r2(X, y):.3f}")
    mc2.metric("Neural Net R²",  f"{nn.r2(X, y):.3f}")
    mc3.metric("RBF R²",         f"{rbf.r2(X, y):.3f}")

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
        permeant = st.selectbox("Permeant", ["Phenol", "M-Cresol", "Glucose"], key="perm_permeant")
        model_name = st.selectbox("Model", ["Regression", "Neural Network", "RBF Interpolation"], key="perm_model")

        st.subheader("Membrane Composition")
        st.caption("Sliders are locked to sum to 100%.")

        # Four sliders that always sum to 100
        if "perm_s1" not in st.session_state:
            st.session_state.perm_s1 = 50
        if "perm_s2" not in st.session_state:
            st.session_state.perm_s2 = 0
        if "perm_c1" not in st.session_state:
            st.session_state.perm_c1 = 50
        if "perm_c2" not in st.session_state:
            st.session_state.perm_c2 = 0

        s1 = st.slider("Sparsa 1 (%)",   0, 100, st.session_state.perm_s1, key="perm_s1_sl")
        remaining_after_s1 = 100 - s1
        s2_max = remaining_after_s1
        s2 = st.slider("Sparsa 2 (%)",   0, s2_max, min(st.session_state.perm_s2, s2_max), key="perm_s2_sl")
        remaining_after_s2 = remaining_after_s1 - s2
        c1_max = remaining_after_s2
        c1 = st.slider("Carbosil 1 (%)", 0, c1_max, min(st.session_state.perm_c1, c1_max), key="perm_c1_sl")
        c2 = remaining_after_s2 - c1
        st.markdown(f"**Carbosil 2: {c2}%** *(auto-calculated to reach 100%)*")

        total = s1 + s2 + c1 + c2
        st.caption(f"Total: {total}%  |  S1={s1}% S2={s2}% C1={c1}% C2={c2}%")

        if st.button("Calculate Permeability", type="primary", use_container_width=True, key="calc_perm"):
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

            st.caption(f"Model: {res['model']}  |  Permeant: {res['permeant']}")

            st.divider()
            st.subheader("Permeation Visualization")
            mol_colors = {"Phenol": "#e74c3c", "M-Cresol": "#9b59b6", "Glucose": "#f39c12"}
            color = mol_colors.get(res["permeant"], "#3498db")
            render_animation(p, res["permeant"], color)

            st.divider()
            st.subheader("Model Comparison")
            p_reg = predict_with_model("Regression",       res["permeant"], res["s1"], res["s2"], res["c1"], res["c2"])
            p_nn  = predict_with_model("Neural Network",   res["permeant"], res["s1"], res["s2"], res["c1"], res["c2"])
            p_rbf = predict_with_model("RBF Interpolation",res["permeant"], res["s1"], res["s2"], res["c1"], res["c2"])
            cmp_df = pd.DataFrame({
                "Model": ["Regression", "Neural Network", "RBF Interpolation"],
                "Permeability (cm/s)": [f"{p_reg:.3e}", f"{p_nn:.3e}", f"{p_rbf:.3e}"],
                "log P": [f"{np.log10(p_reg):.2f}", f"{np.log10(p_nn):.2f}", f"{np.log10(p_rbf):.2f}"]
            })
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Set a composition and click Calculate Permeability.")


# ============== TAB 3: OPTIMAL COMPOSITION ==============
with tab_opt:
    st.title("Optimal Composition Finder")
    st.markdown(
        "Finds the single membrane composition that best minimizes permeability "
        "to **Phenol**, **M-Cresol**, and **Glucose** simultaneously — "
        "targeting minimum passage of all three insulin preservatives."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")
        opt_model = st.selectbox("Model", ["Regression", "Neural Network", "RBF Interpolation"], key="opt_model")

        st.divider()
        st.info(
            "The optimizer searches over all compositions (Sparsa1 + Sparsa2 + Carbosil1 + Carbosil2 = 100%) "
            "to find the formulation that minimizes the combined normalized log-permeability "
            "across Phenol, M-Cresol, and Glucose."
        )

        if st.button("Find Optimal Composition", type="primary", use_container_width=True, key="gen_opt"):
            with st.spinner("Optimizing across all three permeants..."):
                result = find_optimal_combined(opt_model)
                st.session_state.opt_result = result
                st.session_state.opt_settings = {"model": opt_model}

    with col2:
        st.subheader("Optimal Composition")

        if "opt_result" in st.session_state and st.session_state.opt_result:
            res = st.session_state.opt_result
            cfg = st.session_state.opt_settings

            st.markdown(f"**Model used: {cfg['model']}**")
            st.markdown("Composition that minimizes permeability to Phenol, M-Cresol, and Glucose:")

            # Composition metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sparsa 1",   f"{res['Sparsa1']:.1f}%")
            c2.metric("Sparsa 2",   f"{res['Sparsa2']:.1f}%")
            c3.metric("Carbosil 1", f"{res['Carbosil1']:.1f}%")
            c4.metric("Carbosil 2", f"{res['Carbosil2']:.1f}%")

            st.divider()

            # Predicted permeabilities for each permeant
            st.subheader("Predicted Permeabilities at Optimal Composition")
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Phenol (cm/s)",   f"{res['perm_phenol']:.3e}",  f"log P = {res['logp_phenol']:.2f}")
            pc2.metric("M-Cresol (cm/s)", f"{res['perm_mcresol']:.3e}", f"log P = {res['logp_mcresol']:.2f}")
            pc3.metric("Glucose (cm/s)",  f"{res['perm_glucose']:.3e}",  f"log P = {res['logp_glucose']:.2f}")

            st.divider()

            # 3D membrane viewer
            st.subheader("Membrane Structure")
            render_tpu_3dmol(
                int(round(res["Sparsa1"])),
                int(round(res["Sparsa2"])),
                int(round(res["Carbosil1"])),
                int(round(res["Carbosil2"])),
                height=500
            )

            st.caption(
                f"Green = Sparsa (PEG-type) chains  |  "
                f"Blue = Carbosil (PDM/silicone) chains  |  "
                f"Orange = Urethane linkages (URE)"
            )

        else:
            st.info("Click 'Find Optimal Composition' to run the multi-permeant optimizer.")
