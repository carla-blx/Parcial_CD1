# Predicción de Impago de Tarjetas de Crédito - Red Neuronal Perceptrón Multicapa

## Contexto del problema

Este proyecto predice si un cliente de tarjeta de crédito incurrirá en impago en el próximo mes, utilizando datos reales de clientes en Taiwán (abril-septiembre 2005). El objetivo es construir un modelo de clasificación binaria que supere a modelos clásicos, con interpretabilidad y despliegue interactivo.

## Dataset

**Default of Credit Card Clients Dataset** (UCI Machine Learning Repository)

- **Variables**: 25 (demográficas, historial de pagos, montos de facturas y pagos)
- **Observaciones**: 30,000 clientes
- **Variable objetivo**: `default.payment.next.month` (1 = impago, 0 = pago normal)
- **Período**: Abril - Septiembre 2005

Referencia: Lichman, M. (2013). UCI Machine Learning Repository.

## Tecnologías utilizadas

- Python 3.9+
- Pandas, NumPy (manipulación de datos)
- Scikit-learn (preprocesamiento, modelos clásicos)
- TensorFlow/Keras y PyTorch (redes neuronales)
- SHAP (interpretabilidad)
- Streamlit (aplicación interactiva)

## Reproducibilidad

**Semilla global fijada en 20220760** en todos los notebooks y scripts.

