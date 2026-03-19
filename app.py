import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

st.set_page_config(page_title="Taller 1 - Test Diagnóstico", layout="wide", page_icon="🩺")

# Estilos personalizados para darle look llamativo
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Taller 1: Evaluación de Test Diagnóstico")
st.markdown("### Por: *Jabes Rodriguez Enriquez*")
st.markdown("Explora las métricas de diagnóstico cambiando tu Umbral (Threshold) en la barra lateral.")

# Barra lateral interactiva
st.sidebar.header("⚙️ Configuración")
st.sidebar.markdown("Puedes jugar con estos valores:")
seed = st.sidebar.number_input("Semilla Inicial (Seed)", value=85, step=1)
threshold = st.sidebar.slider("Umbral de Decisión (Threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

np.random.seed(seed)
N = 100

# 1. Generar la Data
y_true = np.random.binomial(n=1, p=0.35, size=N)

y_score = np.zeros(N)
for i in range(N):
    if y_true[i] == 1:
        y_score[i] = np.random.uniform(0.4, 1.0)
    else:
        y_score[i] = np.random.uniform(0.0, 0.6)

y_pred = (y_score >= threshold).astype(int)

# Funciones Base
def vp(y_t, y_p): return sum((y_t == 1) & (y_p == 1))
def fp(y_t, y_p): return sum((y_t == 0) & (y_p == 1))
def vn(y_t, y_p): return sum((y_t == 0) & (y_p == 0))
def fn(y_t, y_p): return sum((y_t == 1) & (y_p == 0))

v_p = vp(y_true, y_pred)
f_p = fp(y_true, y_pred)
v_n = vn(y_true, y_pred)
f_n = fn(y_true, y_pred)

def sensibilidad(v_p, f_n): return v_p / (v_p + f_n) if (v_p + f_n) != 0 else 0
def especificidad(v_n, f_p): return v_n / (v_n + f_p) if (v_n + f_p) != 0 else 0
def vpp(v_p, f_p): return v_p / (v_p + f_p) if (v_p + f_p) != 0 else 0
def vpn(v_n, f_n): return v_n / (v_n + f_n) if (v_n + f_n) != 0 else 0
def accuracy(v_p, v_n, f_p, f_n): 
    t = v_p + v_n + f_p + f_n
    return (v_p + v_n) / t if t != 0 else 0

sens = sensibilidad(v_p, f_n)
esp = especificidad(v_n, f_p)
v_p_p = vpp(v_p, f_p)
v_p_n = vpn(v_n, f_n)
acc = accuracy(v_p, v_n, f_p, f_n)

# ================= DASHBOARD LAYOUT =================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Matriz de Confusión")
    st.write(f"Con un umbral de: **{threshold:.2f}**")
    matriz = pd.DataFrame({
        'Enfermo (P)': [v_p, f_p],
        'Sano (N)': [f_n, v_n]
    }, index=['Enfermo (R)', 'Sano (R)'])
    
    st.dataframe(matriz.style.background_gradient(cmap='Purples'), use_container_width=True)

with col2:
    st.subheader("📈 Métricas Diagnósticas")
    # Tarjetas divertidas para métricas
    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)
    
    m1.metric(label="Sensibilidad", value=f"{sens:.1%}", help="VP / (VP + FN)")
    m2.metric(label="Especificidad", value=f"{esp:.1%}", help="VN / (VN + FP)")
    m3.metric(label="Accuracy", value=f"{acc:.1%}", help="(VP + VN) / Total")
    m4.metric(label="VPP (Predictivo Pos)", value=f"{v_p_p:.1%}", help="VP / (VP + FP)")
    m5.metric(label="VPN (Predictivo Neg)", value=f"{v_p_n:.1%}", help="VN / (VN + FN)")

st.divider()

col_roc1, col_roc2 = st.columns(2)

with col_roc1:
    st.subheader("Curva ROC (sklearn)")
    fpr, tpr, th_roc = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, marker='o', label='ROC Model', color='#0078D7')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Azar', color='#aaaaaa')
    ax.set_xlabel('FPR (1 - Especificidad)')
    ax.set_ylabel('TPR (Sensibilidad)')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with col_roc2:
    st.subheader("Cálculo de AUC")
    def auc_trapecios(x, y):
        area = 0.0
        for i in range(1, len(x)):
            dx = x[i] - x[i-1]
            h = (y[i] + y[i-1]) / 2.0
            area += dx * h
        return area
    
    auc_val = auc_trapecios(fpr, tpr)
    st.success(f"**Área Bajo la Curva (AUC) por regla del trapecio:** {auc_val:.4f}")
    
    st.info("""
    💡 **Análisis de Escenarios Clínicos:**
    - **Tamizaje (Enfermedad grave):** Deseamos minimizar Falsos Negativos (FN). Por lo tanto debes usar en la barra lateral un **umbral bajo**. Notarás que la sensibilidad sube a casi al 100%.
    - **Confirmación UCI:** Deseamos minimizar Falsos Positivos (FP). Usa un **umbral muy alto** para subir la especificidad y así nadie que esté sano sea tratado sin razón.
    """)

st.divider()

st.subheader("✨ BONO: Curva ROC Manual")
def curva_roc_manual(y_t, y_s, n_umbrales=100):
    umbrales_test = np.linspace(1, 0, n_umbrales)
    datos = []
    for u in umbrales_test:
        y_p_tmp = (y_s >= u).astype(int)
        v_p_t, f_p_t = vp(y_t, y_p_tmp), fp(y_t, y_p_tmp)
        v_n_t, f_n_t = vn(y_t, y_p_tmp), fn(y_t, y_p_tmp)
        s_tmp = sensibilidad(v_p_t, f_n_t)
        e_tmp = especificidad(v_n_t, f_p_t)
        datos.append({'threshold': u, 'fpr': 1 - e_tmp, 'tpr': s_tmp})
    return pd.DataFrame(datos).sort_values('fpr')

df_buc = curva_roc_manual(y_true, y_score)
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df_buc['fpr'], df_buc['tpr'], marker='x', color='#800080', label='ROC Manual desde cero')
ax2.plot([0,1],[0,1], '--', color='#aaaaaa')
ax2.set_xlabel('FPR')
ax2.set_ylabel('TPR')
ax2.legend()
ax2.grid(alpha=0.3)
st.pyplot(fig2)
