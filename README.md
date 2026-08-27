# Telco Customer Churn Prediction

Sistema predictivo de abandono de clientes (Churn) para la industria de telecomunicaciones. Incluye modelo de Machine Learning, explicabilidad con SHAP, inferencia por lotes y dashboard interactivo con Streamlit.

---

## Descripción del Problema

El **Churn** es el fenómeno por el cual un cliente deja de utilizar los servicios de una empresa. En telecomunicaciones, adquirir un nuevo cliente cuesta entre 5 y 25 veces más que retener uno existente. Este proyecto construye un modelo de clasificación binaria que predice la probabilidad de que un cliente abandone, permitiendo al equipo de marketing tomar acciones preventivas a tiempo.

---

## Arquitectura del Proyecto

```
telco_churn_project/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── data_loader.py      # Carga del dataset
│   ├── quality.py           # Limpieza y calidad de datos
│   └── train.py             # Pipeline de entrenamiento
├── models/
│   └── churn_model.joblib   # Modelo serializado
├── app.py                   # Dashboard Streamlit
└── requirements.txt
```

---

## Metodología (6 Pasos)

### 1. Entender el problema (Enfoque de negocio)
Definimos el objetivo: detectar clientes en riesgo de cancelar su servicio para que el equipo de marketing pueda intervenir proactivamente con ofertas o retención.

### 2. Identificar variables predictoras
Seleccionamos 19 variables clave del comportamiento del cliente:
- **Numéricas:** `tenure` (meses con la empresa), `MonthlyCharges`, `TotalCharges`
- **Categóricas:** tipo de contrato, método de pago, servicio de internet, soporte técnico, y más

### 3. Estructurar la base de datos
Consolidamos la información en una estructura tabular centralizada donde cada fila representa un cliente único y cada columna una de las variables predictoras.

### 4. Garantizar calidad del dato
- Conversión de `TotalCharges` a numérico (coerción de errores a NaN)
- Imputación de nulos con la mediana
- Eliminación de `customerID` (sin valor predictivo)

### 5. Estadística descriptiva
Análisis exploratorio (EDA) con gráficos interactivos: distribución de Churn, correlaciones, histogramas de tenure y monthly charges por estado de abandono.

### 6. Predicción con IA
Pipeline de scikit-learn:
- **Preprocesamiento:** `ColumnTransformer` con `StandardScaler` (numéricas) + `OneHotEncoder` (categóricas)
- **Modelo:** `XGBClassifier` (200 árboles, profundidad máxima 5, learning rate 0.1)
- **Evaluación:** Split 80/20 con estratificación
- **Serialización:** `joblib.dump()` → `models/churn_model.joblib`

---

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Manipulación de datos | pandas, numpy |
| Machine Learning | scikit-learn, xgboost |
| Explicabilidad | shap |
| Serialización | joblib |
| Visualización | plotly, matplotlib |
| Interfaz gráfica | streamlit |

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/richramirezf/telco_customer_churn.git
cd telco_customer_churn

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### Entrenar el modelo

```bash
python src/train.py
```

Esto carga el CSV, aplica la limpieza, entrena el pipeline y genera `models/churn_model.joblib`.

### Ejecutar el dashboard

```bash
python -m streamlit run app.py
```

Abre http://localhost:8501 en tu navegador.

---

## Dashboard: Pestañas Disponibles

### 1. EDA & Calidad
- Vista previa del dataset y estadísticas descriptivas
- Distribución de Churn (gráfico de torta)
- Histogramas de tenure y MonthlyCharges por estado de Churn
- Matriz de correlación entre variables numéricas

### 2. Predicción Interactiva
- Formulario en el sidebar con variables clave: tenure, contract, monthly charges, internet service, payment method, género, senior citizen, partner, dependents
- Predicción binaria (Churn / No Churn) con probabilidad porcentual
- Gráfico de barras de distribución de probabilidades
- **Explicabilidad SHAP:** gráfico de cascada (waterfall) que muestra qué variables impulsaron la predicción para ese cliente específico

### 3. Predicción Masiva (CSV)
- Subida de archivos CSV con datos de clientes
- Ejecución de predicciones por lotes
- Generación de columnas `Churn_Prediction` y `Churn_Probability_%`
- Descarga de resultados como CSV

### 4. Cómo funciona
- Sección de storytelling con 6 expanders que explican la metodología del proyecto a un perfil de negocio no técnico

---

## Variables del Dataset

| Variable | Tipo | Descripción |
|---|---|---|
| `customerID` | Categórica | ID único del cliente (eliminada en limpieza) |
| `gender` | Categórica | Género del cliente |
| `SeniorCitizen` | Numérica | 1 si es adulto mayor, 0 si no |
| `Partner` | Categórica | Tiene pareja |
| `Dependents` | Categórica | Tiene dependientes |
| `tenure` | Numérica | Meses de antigüedad |
| `PhoneService` | Categórica | Tiene servicio telefónico |
| `MultipleLines` | Categórica | Líneas múltiples |
| `InternetService` | Categórica | DSL, Fibra óptica o No |
| `OnlineSecurity` | Categórica | Seguridad en línea |
| `OnlineBackup` | Categórica | Respaldo en línea |
| `DeviceProtection` | Categórica | Protección de dispositivos |
| `TechSupport` | Categórica | Soporte técnico |
| `StreamingTV` | Categórica | Streaming de TV |
| `StreamingMovies` | Categórica | Streaming de películas |
| `Contract` | Categórica | Tipo de contrato |
| `PaperlessBilling` | Categórica | Facturación sin papel |
| `PaymentMethod` | Categórica | Método de pago |
| `MonthlyCharges` | Numérica | Cargo mensual |
| `TotalCharges` | Numérica | Cargo total acumulado |
| `Churn` | Target | 1 = Abandonó, 0 = Permanece |

---

## Licencia

Proyecto académico de Machine Learning aplicado a retención de clientes.
