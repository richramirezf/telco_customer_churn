import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from src.data_loader import load_telco_data
from src.quality import run_quality_pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Telco Customer Churn", layout="wide")


# ── Global UI styling ────────────────────────────────────────────────────────
def apply_clean_ui():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Plus+Jakarta+Sans:wght@600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #042f2e;
        }

        .stButton>button {
            background-color: #0d9488;
            color: white;
            border-radius: 8px;
            border: none;
            box-shadow: 0px 4px 6px -1px rgba(0, 0, 0, 0.1), 0px 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: all 0.2s ease;
        }

        .stButton>button:hover {
            background-color: #0f766e;
            box-shadow: 0px 10px 15px -3px rgba(0, 0, 0, 0.1), 0px 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-1px);
        }

        div[data-testid="stForm"], div[data-testid="stExpander"] {
            background-color: #ffffff !important;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
        }

        div[data-testid="stForm"] p, div[data-testid="stForm"] label,
        div[data-testid="stExpander"] p, div[data-testid="stExpander"] label,
        div[data-testid="stExpander"] li {
            color: #374151 !important;
        }

        div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary p, div[data-testid="stExpander"] summary span {
            color: #042f2e !important;
            font-weight: 600;
        }

        div[data-testid="stExpander"] svg {
            fill: #042f2e !important;
            color: #042f2e !important;
        }
    </style>
    """, unsafe_allow_html=True)


apply_clean_ui()

# ── Load data (cached) ───────────────────────────────────────────────────────
@st.cache_data
def load_clean_data():
    df = load_telco_data()
    df = run_quality_pipeline(df)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


df = load_clean_data()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_eda, tab_pred, tab_batch, tab_how = st.tabs(["📊  EDA & Calidad", "🔮  Predicción Interactiva", "📁  Predicción Masiva (CSV)", "🧠  Cómo funciona"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – EDA & Calidad
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("Exploratory Data Analysis")

    # Dataset preview
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(20), width="stretch")

    # Basic stats
    st.subheader("Estadísticas descriptivas")
    st.write(df.describe())

    # Churn distribution
    st.subheader("Distribución de Churn")
    churn_counts = df["Churn"].value_counts().rename({0: "No", 1: "Yes"})
    fig_churn = px.pie(
        churn_counts,
        values=churn_counts.values,
        names=churn_counts.index,
        title="Proporción de Churn",
        color_discrete_sequence=["#636EFA", "#EF553B"],
    )
    st.plotly_chart(fig_churn, width="stretch")

    # Tenure vs Churn
    st.subheader("Distribución de Tenure por Churn")
    fig_tenure = px.histogram(
        df,
        x="tenure",
        color=df["Churn"].map({0: "No", 1: "Yes"}),
        barmode="overlay",
        opacity=0.65,
        color_discrete_sequence=["#636EFA", "#EF553B"],
        labels={"color": "Churn"},
    )
    st.plotly_chart(fig_tenure, width="stretch")

    # MonthlyCharges vs Churn
    st.subheader("Distribución de MonthlyCharges por Churn")
    fig_monthly = px.histogram(
        df,
        x="MonthlyCharges",
        color=df["Churn"].map({0: "No", 1: "Yes"}),
        barmode="overlay",
        opacity=0.65,
        color_discrete_sequence=["#636EFA", "#EF553B"],
        labels={"color": "Churn"},
    )
    st.plotly_chart(fig_monthly, width="stretch")

    # Correlation heatmap (numeric only)
    st.subheader("Correlación entre variables numéricas")
    numeric_df = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
    corr = numeric_df.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Matriz de correlación",
    )
    st.plotly_chart(fig_corr, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Predicción Interactiva
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.header("Predicción de Churn para un Cliente")

    MODEL_PATH = "models/churn_model.joblib"

    @st.cache_resource
    def load_model():
        return joblib.load(MODEL_PATH)

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("Modelo no encontrado. Ejecuta `python src/train.py` primero.")
        st.stop()

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    st.sidebar.header("Datos del cliente")

    tenure = st.sidebar.slider("Tenure (meses)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Cargo mensual ($)", 18.0, 120.0, 50.0, step=1.0)
    total_charges = tenure * monthly_charges  # approximate

    contract = st.sidebar.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Servicio de internet", ["DSL", "Fiber optic", "No"])
    payment_method = st.sidebar.selectbox(
        "Método de pago",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    gender = st.sidebar.selectbox("Género", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    partner = st.sidebar.selectbox("Pareja", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependientes", ["Yes", "No"])

    # ── Predict ───────────────────────────────────────────────────────────────
    if st.sidebar.button("Predecir"):
        input_df = pd.DataFrame(
            {
                "gender": [gender],
                "SeniorCitizen": [senior_citizen],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": ["Yes"],
                "MultipleLines": ["No"],
                "InternetService": [internet_service],
                "OnlineSecurity": ["No"],
                "OnlineBackup": ["No"],
                "DeviceProtection": ["No"],
                "TechSupport": ["No"],
                "StreamingTV": ["No"],
                "StreamingMovies": ["No"],
                "Contract": [contract],
                "PaperlessBilling": ["Yes"],
                "PaymentMethod": [payment_method],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges],
            }
        )

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ El cliente **ABANDONARÁ** (Churn)")
        else:
            st.success("✅ El cliente **NO abandonará** (No Churn)")

        st.metric(
            label="Probabilidad de Churn",
            value=f"{probability[1] * 100:.1f}%",
        )

        st.subheader("Distribución de probabilidades")
        prob_df = pd.DataFrame(
            {"Resultado": ["No Churn", "Churn"], "Probabilidad": [probability[0], probability[1]]}
        )
        fig_prob = px.bar(
            prob_df,
            x="Resultado",
            y="Probabilidad",
            color="Resultado",
            color_discrete_map={"No Churn": "#636EFA", "Churn": "#EF553B"},
            text_auto=".1%",
        )
        st.plotly_chart(fig_prob, width="stretch")

        # ── SHAP Waterfall ──────────────────────────────────────────────────────
        st.subheader("Explicabilidad SHAP (Waterfall)")
        with st.spinner("Calculando valores SHAP..."):
            preprocessor = model.named_steps["preprocessor"]
            xgb_model = model.named_steps["classifier"]

            X_transformed = preprocessor.transform(input_df)
            raw_names = preprocessor.get_feature_names_out()
            feature_names = [name.split("__", 1)[-1] for name in raw_names]

            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer(X_transformed)
            shap_values.feature_names = list(feature_names)

            fig_shap, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig_shap)
            plt.close(fig_shap)
    else:
        st.info("Ajusta los parámetros en la barra lateral y pulsa **Predecir**.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Predicción Masiva (CSV)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.header("Predicción Masiva por Lotes")

    @st.cache_resource
    def load_model_batch():
        return joblib.load("models/churn_model.joblib")

    try:
        model_batch = load_model_batch()
    except FileNotFoundError:
        st.error("Modelo no encontrado. Ejecuta `python src/train.py` primero.")
        st.stop()

    uploaded_file = st.file_uploader("Sube un archivo CSV con datos de clientes", type=["csv"])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.subheader("Vista previa del archivo cargado")
        st.dataframe(df_batch.head(20), width="stretch")

        if st.button("Ejecutar Predicción Masiva"):
            with st.spinner("Procesando predicciones..."):
                df_clean = df_batch.copy()

                if "customerID" in df_clean.columns:
                    df_clean.drop(columns=["customerID"], inplace=True)

                df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
                df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median())

                predictions = model_batch.predict(df_clean)
                probabilities = model_batch.predict_proba(df_clean)[:, 1]

                df_batch["Churn_Prediction"] = np.where(predictions == 1, "Yes", "No")
                df_batch["Churn_Probability_%"] = (probabilities * 100).round(1)

            st.subheader("Resultados de la predicción")
            st.dataframe(df_batch, width="stretch")

            st.success(f"Predicciones completadas para {len(df_batch)} clientes.")

            csv_output = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar resultados como CSV",
                data=csv_output,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Cómo funciona (Metodología)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_how:
    st.header("🧠 ¿Cómo funciona esta Inteligencia Artificial?")
    st.markdown("Esta herramienta no es magia; es el resultado de un marco de trabajo estructurado. Aquí te explicamos paso a paso cómo transformamos datos crudos en predicciones de negocio:")

    st.markdown("---")

    with st.expander("1. Entender el problema (El enfoque de negocio)", expanded=True):
        st.write('''
        **El objetivo:** Antes de tocar una sola línea de código, definimos qué queríamos resolver.
        En este caso, el reto era detectar qué clientes están a punto de cancelar su servicio (fenómeno conocido como *Churn*). La meta de negocio es identificar este riesgo a tiempo para que el equipo de marketing pueda actuar, ofreciendo alternativas o promociones antes de que el cliente tome la decisión final de irse.
        ''')

    with st.expander("2. Identificar variables predictoras (Buscando pistas)"):
        st.write('''
        **La estrategia:** La Inteligencia Artificial necesita "pistas" para adivinar el futuro.
        Seleccionamos factores clave del comportamiento del cliente: ¿Qué tipo de contrato tienen (mensual o anual)? ¿Cuánto pagan al mes? ¿Cuánto tiempo llevan con la empresa? ¿Tienen soporte técnico contratado? Estas características son las piezas del rompecabezas que alimentan al modelo.
        ''')

    with st.expander("3. Estructurar la base de datos (Los cimientos)"):
        st.write('''
        **La organización:** Recopilamos toda la información histórica dispersa de los clientes y la organizamos en una estructura tabular centralizada.
        Imagínalo como un archivo maestro consolidado donde cada fila es un cliente único y cada columna es una de las "pistas" que identificamos en el paso anterior.
        ''')

    with st.expander("4. Garantizar calidad del dato (Limpiando la casa)"):
        st.write('''
        **La purificación:** Los datos del mundo real siempre vienen con errores, vacíos o formatos incorrectos.
        En esta fase crítica, rellenamos huecos de información (como cargos mensuales no registrados), eliminamos datos inútiles (como el ID del cliente, que no ayuda a predecir nada) y transformamos textos a un lenguaje numérico que la computadora pueda procesar correctamente.
        ''')

    with st.expander("5. Estadística descriptiva (El poder visual)"):
        st.write('''
        **El análisis histórico:** Antes de predecir el futuro, miramos el pasado.
        En esta etapa cruzamos las variables para visualizar el panorama general y confirmar hipótesis. Por ejemplo, al graficar los datos, confirmamos visualmente que los clientes que pagan mes a mes tienen una tasa de abandono drásticamente mayor que aquellos con contratos anuales.
        ''')

    with st.expander("6. Predicción con IA (El motor predictivo)"):
        st.write('''
        **El aprendizaje:** Aquí entra en acción el Machine Learning. Usamos un algoritmo avanzado que analizó miles de perfiles de clientes del pasado, aprendiendo de forma autónoma las reglas y patrones ocultos de los que decidieron irse.
        Hoy, cuando evalúas a un cliente nuevo en esta aplicación, la IA compara su perfil con todo lo que aprendió y emite un veredicto matemático: la probabilidad exacta de abandono.
        ''')
