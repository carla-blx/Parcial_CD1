# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:45:17 2026

@author: LENOVO
"""
# app.py
# =============================================================================
# STREAMLIT APP - PREDICCIÓN DE DEFAULT
# ============================================================================

import streamlit as st
st.set_page_config(
    page_title="Predicción de Default",
    page_icon="🚨",
    layout="wide"
)

# Resto de imports DESPUÉS de st.set_page_config
import pandas as pd
import plotly.graph_objects as go
import joblib
import pickle
import numpy as np

from utils import (
    load_pipeline,
    load_keras_model,
    load_sklearn_model,
    preprocess_input,
    predict_keras,
    predict_sklearn,
    get_model_info
)

# =============================================================================
# CSS PERSONALIZADO
# =============================================================================

def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Si no existe style.css, no hacer nada
        pass

# =============================================================================
# CARGA DE MODELOS
# =============================================================================

@st.cache_resource
def load_resources():
    pipeline = load_pipeline()
    keras_model = load_keras_model()
    sklearn_model = load_sklearn_model()
    
    # Verificar si los modelos se cargaron correctamente
    if keras_model is None:
        st.warning("⚠️ Modelo Keras no disponible. Usando valores por defecto.")
    if sklearn_model is None:
        st.warning("⚠️ Modelo Sklearn no disponible. Usando valores por defecto.")
    
    return pipeline, keras_model, sklearn_model

pipeline, keras_model, sklearn_model = load_resources()
    
# =============================================================================
# HEADER
# =============================================================================

st.title("🚨 Predicción de Riesgo de Default")

st.markdown("""
Sistema de predicción de incumplimiento crediticio utilizando:

- Deep Learning con Keras/TensorFlow
- Redes neuronales MLP con Scikit-Learn
- Ingeniería de variables
- Modelos entrenados sobre datos financieros reales
""")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown("## ⚙️ Configuración")

    model_choice = st.selectbox(
        "Selecciona el modelo",
        ["Keras", "Scikit-Learn"]
    )

    st.markdown("---")

    st.markdown("## 📊 Estado Modelos")

    model_info = get_model_info()

    st.markdown("### Keras")
    st.success(model_info['Keras']['status'])

    st.markdown("### Scikit-Learn")
    st.success(model_info['Scikit-Learn']['status'])

    st.markdown("---")

    st.markdown("## ℹ️ Variables")

    st.caption("""
    Dataset de tarjetas de crédito:
    
    - Historial de pagos
    - Montos facturados
    - Pagos realizados
    - Datos demográficos
    """)

# =============================================================================
# FORMULARIO
# =============================================================================

st.markdown("## 📝 Datos del Cliente")

# =============================================================================
# FILA 1
# =============================================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    st.markdown("### 👤 Datos Generales")

    LIMIT_BAL = st.number_input(
        "Límite de crédito",
        min_value=10000,
        max_value=1000000,
        value=200000,
        step=10000
    )

    AGE = st.slider(
        "Edad",
        18,
        80,
        30
    )

    SEX = st.selectbox(
        "Sexo",
        [1, 2],
        format_func=lambda x: "Masculino" if x == 1 else "Femenino"
    )

    EDUCATION = st.selectbox(
        "Educación",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Posgrado",
            2: "Universidad",
            3: "Secundaria",
            4: "Otros"
        }[x]
    )

    MARRIAGE = st.selectbox(
        "Estado civil",
        [1, 2, 3],
        format_func=lambda x: {
            1: "Casado",
            2: "Soltero",
            3: "Otros"
        }[x]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FILA 2
# =============================================================================

with col2:

    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    st.markdown("### 💳 Historial de Pago")

    pay_help = """
    -2 = Sin consumo
    -1 = Pago puntual
     0 = Uso normal
     1 = 1 mes de atraso
     2 = 2 meses de atraso
     3 = 3 meses de atraso
     4 = 4 meses de atraso
     5 = 5 meses de atraso
     6 = 6 meses de atraso
     7 = 7 meses de atraso
     8 = 8 meses o más
    """
    
    PAY_1 = st.slider(
        "Estado de pago - Septiembre 2005",
        -2,
        8,
        0,
        help=pay_help
    )
    
    PAY_2 = st.slider(
        "Estado de pago - Agosto 2005",
        -2,
        8,
        0,
        help=pay_help
    )
    
    PAY_3 = st.slider(
        "Estado de pago - Julio 2005",
        -2,
        8,
        0,
        help=pay_help
    )
    
    PAY_4 = st.slider(
        "Estado de pago - Junio 2005",
        -2,
        8,
        0,
        help=pay_help
    )
    
    PAY_5 = st.slider(
        "Estado de pago - Mayo 2005",
        -2,
        8,
        0,
        help=pay_help
    )
    
    PAY_6 = st.slider(
        "Estado de pago - Abril 2005",
        -2,
        8,
        0,
        help=pay_help
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FILA 3
# =============================================================================

with col3:

    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    st.markdown("### 💰 Facturación")

    BILL_AMT1 = st.number_input("Monto facturado - Septiembre", value=5000)
    BILL_AMT2 = st.number_input("Monto facturado - Agosto", value=4000)
    BILL_AMT3 = st.number_input("Monto facturado - Julio", value=3000)
    BILL_AMT4 = st.number_input("Monto facturado - Junio", value=2500)
    BILL_AMT5 = st.number_input("Monto facturado - Mayo", value=2000)
    BILL_AMT6 = st.number_input("Monto facturado - Abril", value=1500)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FILA 4
# =============================================================================

st.markdown("## 💵 Pagos Realizados")

col4, col5, col6 = st.columns(3)

with col4:

    PAY_AMT1 = st.number_input("Pago realizado - Septiembre", value=1000)
    PAY_AMT2 = st.number_input("Pago realizado - Agosto", value=1000)

with col5:

    PAY_AMT3 = st.number_input("Pago realizado - Julio", value=1000)
    PAY_AMT4 = st.number_input("Pago realizado - Junio", value=1000)

with col6:

    PAY_AMT5 = st.number_input("Pago realizado - Mayo", value=1000)
    PAY_AMT6 = st.number_input("Pago realizado - Abril", value=1000)

# =============================================================================
# BOTÓN
# =============================================================================

st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])

with col_btn2:

    predict_button = st.button(
        "🔮 PREDECIR",
        use_container_width=True
    )

# =============================================================================
# PREDICCIÓN
# =============================================================================

if predict_button:
    input_data = {
        'LIMIT_BAL': LIMIT_BAL,
        'AGE': AGE,
        'PAY_1': PAY_1,
        'PAY_2': PAY_2,
        'PAY_3': PAY_3,
        'PAY_4': PAY_4,
        'PAY_5': PAY_5,
        'PAY_6': PAY_6,
        'BILL_AMT1': BILL_AMT1,
        'BILL_AMT2': BILL_AMT2,
        'BILL_AMT3': BILL_AMT3,
        'BILL_AMT4': BILL_AMT4,
        'BILL_AMT5': BILL_AMT5,
        'BILL_AMT6': BILL_AMT6,
        'PAY_AMT1': PAY_AMT1,
        'PAY_AMT2': PAY_AMT2,
        'PAY_AMT3': PAY_AMT3,
        'PAY_AMT4': PAY_AMT4,
        'PAY_AMT5': PAY_AMT5,
        'PAY_AMT6': PAY_AMT6,
        'SEX': SEX,
        'EDUCATION': EDUCATION,
        'MARRIAGE': MARRIAGE
    }
    
    with st.spinner("Procesando datos y generando predicción..."):
        # Preprocesar usando la nueva función
        X_processed = preprocess_input(input_data)
        
        if model_choice == "Keras":
            proba = predict_keras(keras_model, X_processed)
            model_used = "Keras"
        else:
            proba = predict_sklearn(sklearn_model, X_processed)
            model_used = "Scikit-Learn"

    # =========================================================
    # RESULTADO
    # =========================================================

    if proba >= 0.5:

        clase = "ALTO RIESGO DE DEFAULT 🔴"

        descripcion = """
        El cliente presenta alta probabilidad de incumplimiento.
        """

        color_clase = "#C73E1D"

    else:

        clase = "BAJO RIESGO DE DEFAULT 🟢"

        descripcion = """
        El cliente presenta bajo riesgo de incumplimiento.
        """

        color_clase = "#2E8B57"

    # ========================================================
    # TARJETA RESULTADO
    # =========================================================

    st.markdown(f"""
<div class="result-card">

<h2>📊 Resultado de Predicción</h2>

<p><b>Probabilidad:</b> {proba:.2%}</p>

<p><b>Clase:</b> {clase}</p>

<p><b>Modelo:</b> {model_used}</p>

<p>{descripcion}</p>

</div>
""", unsafe_allow_html=True)

    # =========================================================
    # INTERPRETABILIDAD
    # =========================================================

    with st.expander("🔍 Variables más importantes (SHAP)"):

        features = [
            'meses_mora',
            'log_limit_bal',
            'PAY_1',
            'PAY_2',
            'LIMIT_BAL',
            'AGE'
        ]

        shap_values = [
            0.10,
            0.056,
            0.033,
            0.027,
            0.025,
            0.018
        ]

        fig = go.Figure(go.Bar(

            x=shap_values,
            y=features,
            orientation='h',

            marker_color='#2E86AB',

            text=[f"{v:.3f}" for v in shap_values],

            textposition='outside'

        ))

        fig.update_layout(

            title="Importancia de Variables",

            xaxis_title="Contribución SHAP",

            yaxis_title="Variables",

            height=400

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =============================================================================
# MÉTRICAS
# =============================================================================

st.markdown("---")

st.markdown("## 📈 Rendimiento del mejor modelo")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)

    st.metric(
        "Accuracy",
        "75.9%",
        "+1.6%"
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col_m2:

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)

    st.metric(
        "ROC-AUC",
        "0.764",
        "+0.02"
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col_m3:

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)

    st.metric(
        "F1-Score",
        "0.513",
        "+0.01"
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col_m4:

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)

    st.metric(
        "Modelo",
        "Keras",
        "Mejor desempeño"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div class="footer">

<p>
 Examen Parcial
</p>

<p style="font-size:0.8rem;">
TensorFlow/Keras + Scikit-Learn + Streamlit
</p>

</div>
""", unsafe_allow_html=True)
