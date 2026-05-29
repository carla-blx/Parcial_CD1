# utils.py - Versión corregida definitiva
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pickle
import streamlit as st  #

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
        
        print(f"✅ Pipeline cargado correctamente")
        print(f"   Columnas numéricas: {len(NUM_COLS)}")
        print(f"   Columnas categóricas: {len(CAT_COLS)}")
        print(f"   Features finales: {len(FEATURES_FINALES)}")
        
        return PIPELINE
    except Exception as e:
        print(f"❌ Error cargando pipeline: {e}")
        return None

# Cargar pipeline al inicio
#load_pipeline()

# =============================================================================
# FUNCIONES DE CREACIÓN DE FEATURES
# =============================================================================

def crear_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las 5 features derivadas exactamente como en el entrenamiento
    """
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
    """
    Preprocesa los datos usando el pipeline real.
    El pipeline espera: NUM_COLS (25) + CAT_COLS (3) = 28 columnas de entrada
    """
    
    if PIPELINE is None:
        raise ValueError("No se pudo cargar el pipeline")
    
    # 1. Crear DataFrame con todos los datos
    df = pd.DataFrame([data_dict])
    
    print(f"Columnas iniciales: {df.columns.tolist()}")
    
    # 2. Aplicar mapeos categóricos
    df = aplicar_mapeos_categoricos(df)
    
    # 3. Crear features derivadas (esto agrega 5 nuevas columnas)
    df = crear_features_derivadas(df)
    
    print(f"Después de features derivadas: {df.columns.tolist()}")
    
    # 4. IMPORTANTE: El pipeline espera las columnas en el orden: NUM_COLS + CAT_COLS
    # NUM_COLS ya incluye las 5 features derivadas
    columnas_esperadas = NUM_COLS + CAT_COLS
    
    print(f"Columnas esperadas por pipeline: {len(columnas_esperadas)}")
    print(f"  Primeras 5 numéricas: {NUM_COLS[:5]}")
    print(f"  Categóricas: {CAT_COLS}")
    
    # 5. Verificar que todas las columnas existan
    for col in columnas_esperadas:
        if col not in df.columns:
            print(f"⚠️ Columna faltante: {col}, agregando con valor 0")
            df[col] = 0
    
    # 6. Seleccionar solo las columnas esperadas en el orden correcto
    df_final = df[columnas_esperadas]
    
    print(f"Shape final antes de pipeline: {df_final.shape}")
    print(f"Columnas: {df_final.columns.tolist()[:5]}... + {CAT_COLS}")
    
    # 7. Aplicar el pipeline
    try:
        X_processed = PIPELINE.transform(df_final)
        print(f"✅ Preprocesamiento exitoso. Shape: {X_processed.shape}")
        return X_processed.astype(np.float32)
    except Exception as e:
        print(f"❌ Error en pipeline.transform: {e}")
        print(f"   DataFrame shape: {df_final.shape}")
        print(f"   DataFrame columns: {df_final.columns.tolist()}")
        raise

# =============================================================================
# LOADERS
# =============================================================================

# utils.py - Reemplaza la función load_keras_model

def load_keras_model():
    """Carga modelo desde JSON y pesos - COMPATIBLE 100%"""
    import json
    import os
    
    try:
        # Buscar archivos JSON y pesos en diferentes ubicaciones
        json_file = None
        weights_file = None
        
        # Lista de posibles ubicaciones
        posibles_ubicaciones = [
            ("model_architecture.json", "model.weights.h5"),
            ("modelo_final/model_architecture.json", "modelo_final/model_weights.h5"),
            ("data/model_architecture.json", "data/model_weights.h5"),
            ("../model_architecture.json", "../model_weights.h5"),
        ]
        
        for json_path, weights_path in posibles_ubicaciones:
            if os.path.exists(json_path) and os.path.exists(weights_path):
                json_file = json_path
                weights_file = weights_path
                break
        
        # Si no encuentra los archivos, mostrar error claro
        if json_file is None:
            error_msg = """
            ❌ No se encontraron los archivos del modelo.
            
            Archivos necesarios:
            - model_architecture.json
            - model.weights.h5
            
            Por favor, asegúrate de haber subido estos archivos a tu repositorio.
            """
            st.error(error_msg)
            return None
        
        # Cargar arquitectura desde JSON
        with open(json_file, 'r') as f:
            model_json = f.read()
        
        # Reconstruir el modelo (sin compilar)
        model = tf.keras.models.model_from_json(model_json)
        
        # Cargar los pesos
        model.load_weights(weights_file)
        
        # Compilar el modelo (ajusta según tu problema)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',  # Cambia si es clasificación multiclase
            metrics=['accuracy']
        )
        
        st.success(f"✅ Modelo Keras cargado exitosamente desde {json_file}")
        return model
        
    except Exception as e:
        st.error(f"❌ Error detallado al cargar modelo: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def load_sklearn_model():
    """Carga el modelo de Scikit-Learn"""
    model = joblib.load("sklearn_model.pkl")
    return model

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
    proba = model.predict(X, verbose=0)[0][0]
    return float(proba)

def predict_sklearn(model, X: np.ndarray) -> float:
    """Predicción con Scikit-Learn"""
    proba = model.predict_proba(X)[0][1]
    return float(proba)
