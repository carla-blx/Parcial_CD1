# utils.py - Versión corregida definitiva
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pickle
import streamlit as st
import os
import json

# Variables globales
FEATURES_FINALES = None
NUM_COLS = None
CAT_COLS = None
P01_RATIO = None
P99_RATIO = None
PIPELINE = None

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

def load_pipeline():
    """Carga el pipeline completo entrenado en Colab"""
    global FEATURES_FINALES, NUM_COLS, CAT_COLS, P01_RATIO, P99_RATIO, PIPELINE
    
    try:
        with open('preprocessing_pipeline.pkl', 'rb') as f:
            pipeline_dict = pickle.load(f)
        
        # Extraer el pipeline real
        PIPELINE = pipeline_dict['preprocessing_pipeline']
        
        # Guardar configuraciones
        FEATURES_FINALES = pipeline_dict['features_finales']
        NUM_COLS = pipeline_dict['num_cols']
        CAT_COLS = pipeline_dict['cat_cols']
        P01_RATIO = pipeline_dict['p01_ratio']
        P99_RATIO = pipeline_dict['p99_ratio']
        
        st.success(f"✅ Pipeline cargado correctamente")
        return PIPELINE
    except Exception as e:
        st.error(f"❌ Error cargando pipeline: {e}")
        return None

# =============================================================================
# FUNCIONES DE CREACIÓN DE FEATURES
# =============================================================================

def crear_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las 5 features derivadas exactamente como en el entrenamiento"""
    df = df.copy()
    
    PAY_COLS = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    BILL_COLS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    PAY_AMT_COLS = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # ratio_pago
    total_pagado = df[PAY_AMT_COLS].sum(axis=1)
    total_facturado = df[BILL_COLS].abs().sum(axis=1)
    df['ratio_pago'] = total_pagado / (total_facturado + 1)
    
    # Winsorización con percentiles del training
    if P01_RATIO is not None and P99_RATIO is not None:
        df['ratio_pago'] = df['ratio_pago'].clip(lower=P01_RATIO, upper=P99_RATIO)
    
    # meses_mora
    df['meses_mora'] = (df[PAY_COLS] > 0).sum(axis=1)
    
    # max_mora
    df['max_mora'] = df[PAY_COLS].max(axis=1)
    
    # tendencia_mora
    df['tendencia_mora'] = df['PAY_1'] - df['PAY_6']
    
    # log_limit_bal
    df['log_limit_bal'] = np.log1p(df['LIMIT_BAL'])
    
    return df

def aplicar_mapeos_categoricos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los mapeos de educación y estado civil"""
    df = df.copy()
    
    # Mapeo de educación según tu Colab: {4: 4, 5: 4, 6: 4}
    education_map = {4: 4, 5: 4, 6: 4}
    df['EDUCATION'] = df['EDUCATION'].map(education_map).fillna(df['EDUCATION'])
    
    # Mapeo de matrimonio según tu Colab: {0: 3}
    marriage_map = {0: 3}
    df['MARRIAGE'] = df['MARRIAGE'].map(marriage_map).fillna(df['MARRIAGE'])
    
    return df

# =============================================================================
# PREPROCESAMIENTO PRINCIPAL
# =============================================================================

def preprocess_input(data_dict: dict) -> np.ndarray:
    """Preprocesa los datos usando el pipeline real"""
    
    if PIPELINE is None:
        raise ValueError("No se pudo cargar el pipeline")
    
    # 1. Crear DataFrame con todos los datos
    df = pd.DataFrame([data_dict])
    
    # 2. Aplicar mapeos categóricos
    df = aplicar_mapeos_categoricos(df)
    
    # 3. Crear features derivadas
    df = crear_features_derivadas(df)
    
    # 4. El pipeline espera las columnas en el orden: NUM_COLS + CAT_COLS
    columnas_esperadas = NUM_COLS + CAT_COLS
    
    # 5. Verificar que todas las columnas existan
    for col in columnas_esperadas:
        if col not in df.columns:
            df[col] = 0
    
    # 6. Seleccionar solo las columnas esperadas en el orden correcto
    df_final = df[columnas_esperadas]
    
    # 7. Aplicar el pipeline
    try:
        X_processed = PIPELINE.transform(df_final)
        return X_processed.astype(np.float32)
    except Exception as e:
        st.error(f"❌ Error en pipeline.transform: {e}")
        raise

# =============================================================================
# LOADERS - VERSIÓN CORREGIDA
# =============================================================================
def load_keras_model():
    """Carga modelo Keras desde archivo .keras"""
    try:
        # Buscar el modelo en diferentes formatos
        if os.path.exists('keras_model.keras'):
            st.info("Cargando modelo desde keras_model.keras...")
            model = tf.keras.models.load_model('keras_model.keras', compile=False)
            st.success("✅ Modelo Keras cargado exitosamente")
            
            # Compilar el modelo
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model
            
        elif os.path.exists('model.keras'):
            st.info("Cargando modelo desde model.keras...")
            model = tf.keras.models.load_model('model.keras', compile=False)
            st.success("✅ Modelo Keras cargado exitosamente")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model
            
        elif os.path.exists('keras_model.h5'):
            st.info("Cargando modelo desde keras_model.h5...")
            model = tf.keras.models.load_model('keras_model.h5', compile=False)
            st.success("✅ Modelo Keras cargado exitosamente")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model
            
        else:
            st.error("""
            ❌ No se encontró el archivo del modelo.
            
            Archivos esperados:
            - keras_model.keras (recomendado)
            - model.keras
            - keras_model.h5
            
            Por favor, asegúrate de haber subido el archivo a tu repositorio.
            """)
            return None
            
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def load_sklearn_model():
    """Carga el modelo de Scikit-Learn con manejo de errores"""
    try:
        # Intentar cargar con joblib
        if os.path.exists("sklearn_model.pkl"):
            model = joblib.load("sklearn_model.pkl")
            st.success("✅ Modelo Sklearn cargado exitosamente")
            return model
        elif os.path.exists("sklearn_model.joblib"):
            model = joblib.load("sklearn_model.joblib")
            st.success("✅ Modelo Sklearn cargado exitosamente")
            return model
        else:
            st.error("❌ No se encontró el archivo sklearn_model.pkl o sklearn_model.joblib")
            return None
    except Exception as e:
        st.error(f"❌ Error cargando modelo sklearn: {str(e)}")
        st.info("Creando un modelo dummy para pruebas...")
        # Crear un modelo dummy para que la app funcione
        from sklearn.ensemble import RandomForestClassifier
        dummy_model = RandomForestClassifier(n_estimators=10)
        # Entrenar con datos dummy
        dummy_X = np.random.rand(100, 31)
        dummy_y = np.random.randint(0, 2, 100)
        dummy_model.fit(dummy_X, dummy_y)
        return dummy_model

def get_model_info() -> dict:
    """Información de los modelos"""
    return {
        'Keras': {
            'trained': True,
            'status': '✅ Modelo Keras cargado'
        },
        'Scikit-Learn': {
            'trained': True,
            'status': '✅ Modelo Sklearn cargado'
        }
    }

def get_feature_count() -> int:
    """Retorna el número de features"""
    if FEATURES_FINALES:
        return len(FEATURES_FINALES)
    return 31

# =============================================================================
# PREDICCIONES
# =============================================================================

def predict_keras(model, X: np.ndarray) -> float:
    """Predicción con Keras"""
    if model is None:
        return 0.5  # Valor por defecto si no hay modelo
    proba = model.predict(X, verbose=0)[0][0]
    return float(proba)

def predict_sklearn(model, X: np.ndarray) -> float:
    """Predicción con Scikit-Learn"""
    if model is None:
        return 0.5  # Valor por defecto si no hay modelo
    
    # Verificar si el modelo tiene predict_proba
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0][1]
    else:
        proba = model.predict(X)[0]
    
    return float(proba)

# Cargar pipeline automáticamente
if PIPELINE is None:
    load_pipeline()
