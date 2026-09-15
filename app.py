import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Carbon Emission AI Analysis",
    page_icon="🌍",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_dataset():
    return pd.read_csv("ml_ready_dataset.csv")


@st.cache_data
def load_cleaned_dataset():
    return pd.read_csv("cleaned_dataset.csv")


df = load_dataset()
cleaned_df = load_cleaned_dataset()


# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models():

    features = [
        "YEAR",
        "YEARS_SINCE_1750",
        "YEAR_SQUARED",
        "COUNTRY"
    ]

    X = df[features]
    y = df["OBS_VALUE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "country",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                ["COUNTRY"]
            )
        ],
        remainder="passthrough"
    )

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "SVR": SVR(
            kernel="rbf",
            C=100,
            epsilon=0.1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror"
        )
    }

    trained_models = {}
    results = []

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        })

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="R2 Score",
        ascending=False
    ).reset_index(drop=True)

    # XGBoost is the primary model for this project
    primary_model = trained_models["XGBoost"]

    return (
        trained_models,
        results_df,
        primary_model,
        X_test,
        y_test
    )


with st.spinner("Preparing AI models..."):

    (
        trained_models,
        results_df,
        model,
        X_test,
        y_test
    ) = train_models()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def predict_co2(country, year):

    input_data = pd.DataFrame({
        "YEAR": [year],
        "YEARS_SINCE_1750": [year - 1750],
        "YEAR_SQUARED": [year ** 2],
        "COUNTRY": [country]
    })

    prediction = model.predict(input_data)[0]

    return prediction


def classify_emission(value):

    if value < 1:
        return "Low"

    elif value < 10:
        return "Moderate"

    elif value < 100:
        return "High"

    else:
        return "Very High"


def get_shap_explanation(country, year):

    input_data = pd.DataFrame({
        "YEAR": [year],
        "YEARS_SINCE_1750": [year - 1750],
        "YEAR_SQUARED": [year ** 2],
        "COUNTRY": [country]
    })

    xgb_model = model.named_steps["model"]

    preprocessor = model.named_steps["preprocessor"]

    transformed_data = preprocessor.transform(
        input_data
    )

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = explainer.shap_values(
        transformed_data
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0]
    })

    shap_df["Importance"] = (
        shap_df["SHAP Value"].abs()
    )

    useful_features = []

    for _, row in shap_df.iterrows():

        feature = row["Feature"]

        if "COUNTRY_" in feature:

            index = list(
                feature_names
            ).index(feature)

            if transformed_data[0][index] != 0:

                useful_features.append(row)

        else:

            useful_features.append(row)

    if len(useful_features) > 0:

        shap_df = pd.DataFrame(
            useful_features
        )

    shap_df = shap_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    return shap_df.head(5), shap_df


def generate_recommendations(emission_level):

    if emission_level == "Low":

        return [
            "Continue monitoring CO₂ emission trends.",
            "Maintain energy-efficient practices.",
            "Increase the use of renewable energy sources.",
            "Promote sustainable transportation and low-carbon technologies."
        ]

    elif emission_level == "Moderate":

        return [
            "Improve energy efficiency in major energy-consuming activities.",
            "Increase renewable energy adoption.",
            "Reduce unnecessary energy consumption.",
            "Promote low-carbon transportation and technologies."
        ]

    elif emission_level == "High":

        return [
            "Prioritize energy-efficiency improvements.",
            "Increase the use of renewable and low-carbon energy sources.",
            "Reduce dependence on high-carbon energy sources.",
            "Promote energy-efficient transportation and technologies.",
            "Continuously monitor future CO₂ emission trends."
        ]

    else:

        return [
            "Immediately prioritize major carbon-reduction measures.",
            "Strongly increase renewable-energy adoption.",
            "Reduce dependence on carbon-intensive energy sources.",
            "Improve energy efficiency across major activities.",
            "Promote low-carbon transportation and technologies.",
            "Implement continuous CO₂ monitoring and reduction planning."
        ]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌍 Carbon Emission AI")

st.sidebar.write(
    "AI-Based Analysis of Carbon Emission "
    "Using Machine Learning Techniques"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Data Analysis",
        "CO₂ Prediction",
        "SHAP Explanation",
        "Recommendations",
        "Previous vs Present",
        "Model Comparison"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("🌍 Carbon Emission AI Dashboard")

    st.write(
        "AI-based carbon emission analysis using "
        "machine learning and explainable AI."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Countries / Areas",
            df["COUNTRY"].nunique()
        )

    with col3:
        st.metric(
            "Years",
            f"{df['YEAR'].min()} - {df['YEAR'].max()}"
        )

    with col4:
        st.metric(
            "Primary Model",
            "XGBoost"
        )

    st.divider()

    st.subheader("Quick CO₂ Prediction")

    countries = sorted(
        df["COUNTRY"].dropna().unique()
    )

    selected_country = st.selectbox(
        "Select Country",
        countries,
        index=(
            countries.index("India")
            if "India" in countries
            else 0
        )
    )

    selected_year = st.number_input(
        "Select Year",
        min_value=int(df["YEAR"].min()),
        max_value=2100,
        value=2023,
        step=1
    )

    prediction = predict_co2(
        selected_country,
        selected_year
    )

    level = classify_emission(
        prediction
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Predicted CO₂",
            f"{prediction:.3f}"
        )

    with c2:

        st.metric(
            "Emission Level",
            level
        )

    st.info(
        "The prediction is generated using "
        "XGBoost based on country and year-related "
        "features available in the dataset."
    )


# =========================================================
# DATA ANALYSIS
# =========================================================

elif page == "Data Analysis":

    st.title("📊 Data Analysis")

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Columns",
            len(df.columns)
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.divider()

    st.subheader("CO₂ Emission Distribution")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        df["OBS_VALUE"],
        bins=50
    )

    ax.set_xlabel(
        "CO₂ Emission"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_title(
        "Distribution of CO₂ Emissions"
    )

    st.pyplot(fig)

    st.divider()

    st.subheader(
        "Top 10 Countries by Average CO₂"
    )

    country_avg = (
        df.groupby("COUNTRY")["OBS_VALUE"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    st.bar_chart(
        country_avg
    )


# =========================================================
# CO2 PREDICTION
# =========================================================

elif page == "CO₂ Prediction":

    st.title("🔮 CO₂ Emission Prediction")

    countries = sorted(
        df["COUNTRY"].dropna().unique()
    )

    country = st.selectbox(
        "Select Country",
        countries,
        index=(
            countries.index("India")
            if "India" in countries
            else 0
        )
    )

    year = st.number_input(
        "Enter Year",
        min_value=int(df["YEAR"].min()),
        max_value=2100,
        value=2023,
        step=1
    )

    if st.button(
        "Predict CO₂ Emission",
        type="primary"
    ):

        prediction = predict_co2(
            country,
            year
        )

        level = classify_emission(
            prediction
        )

        st.success(
            "Prediction completed successfully!"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted CO₂",
                f"{prediction:.3f}"
            )

        with col2:

            st.metric(
                "Emission Level",
                level
            )

        st.divider()

        st.subheader(
            "Prediction Interpretation"
        )

        if level == "Low":

            st.success(
                "The predicted emission level is Low."
            )

        elif level == "Moderate":

            st.warning(
                "The predicted emission level is Moderate."
            )

        elif level == "High":

            st.warning(
                "The predicted emission level is High. "
                "Carbon-reduction measures should be prioritized."
            )

        else:

            st.error(
                "The predicted emission level is Very High. "
                "Immediate carbon-reduction measures are recommended."
            )


# =========================================================
# SHAP EXPLANATION
# =========================================================

elif page == "SHAP Explanation":

    st.title("🔍 Explainable AI - SHAP")

    st.write(
        "SHAP helps explain how the model features "
        "contribute to the prediction."
    )

    countries = sorted(
        df["COUNTRY"].dropna().unique()
    )

    country = st.selectbox(
        "Select Country",
        countries,
        index=(
            countries.index("India")
            if "India" in countries
            else 0
        ),
        key="shap_country"
    )

    year = st.number_input(
        "Select Year",
        min_value=int(df["YEAR"].min()),
        max_value=2100,
        value=2023,
        key="shap_year"
    )

    if st.button(
        "Analyze Prediction",
        type="primary"
    ):

        prediction = predict_co2(
            country,
            year
        )

        level = classify_emission(
            prediction
        )

        top_shap, all_shap = (
            get_shap_explanation(
                country,
                year
            )
        )

        st.metric(
            "Predicted CO₂",
            f"{prediction:.3f}"
        )

        st.metric(
            "Emission Level",
            level
        )

        st.divider()

        st.subheader(
            "Top Contributing Features"
        )

        display_shap = top_shap[
            ["Feature", "SHAP Value"]
        ].copy()

        display_shap["Effect"] = (
            display_shap["SHAP Value"]
            .apply(
                lambda x:
                "Increases prediction"
                if x > 0
                else "Decreases prediction"
            )
        )

        st.dataframe(
            display_shap,
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "SHAP Feature Importance"
        )

        chart_df = (
            top_shap
            .set_index("Feature")
            ["Importance"]
        )

        st.bar_chart(
            chart_df
        )

        st.info(
            "SHAP explains model behavior. "
            "A SHAP contribution should not be interpreted "
            "as proof of a real-world causal relationship."
        )


# =========================================================
# RECOMMENDATIONS
# =========================================================

elif page == "Recommendations":

    st.title("💡 Carbon Reduction Recommendations")

    countries = sorted(
        df["COUNTRY"].dropna().unique()
    )

    country = st.selectbox(
        "Select Country",
        countries,
        index=(
            countries.index("India")
            if "India" in countries
            else 0
        )
    )

    year = st.number_input(
        "Select Year",
        min_value=int(df["YEAR"].min()),
        max_value=2100,
        value=2023
    )

    prediction = predict_co2(
        country,
        year
    )

    level = classify_emission(
        prediction
    )

    st.metric(
        "Predicted CO₂",
        f"{prediction:.3f}"
    )

    st.metric(
        "Emission Level",
        level
    )

    st.divider()

    st.subheader(
        "Recommended Actions"
    )

    recommendations = (
        generate_recommendations(
            level
        )
    )

    for i, recommendation in enumerate(
        recommendations,
        start=1
    ):

        st.write(
            f"**{i}.** {recommendation}"
        )

    st.divider()

    st.info(
        "Recommendations are generated according "
        "to the predicted emission level. "
        "The current dataset does not contain "
        "building-level energy, occupancy or "
        "temperature variables."
    )


# =========================================================
# PREVIOUS VS PRESENT
# =========================================================

elif page == "Previous vs Present":

    st.title("📈 Previous vs Present CO₂")

    countries = sorted(
        df["COUNTRY"].dropna().unique()
    )

    country = st.selectbox(
        "Select Country",
        countries,
        index=(
            countries.index("India")
            if "India" in countries
            else 0
        )
    )

    country_data = (
        df[
            df["COUNTRY"] == country
        ]
        .sort_values("YEAR")
    )

    if len(country_data) >= 2:

        previous_row = (
            country_data.iloc[-2]
        )

        latest_row = (
            country_data.iloc[-1]
        )

        previous_value = (
            previous_row["OBS_VALUE"]
        )

        latest_value = (
            latest_row["OBS_VALUE"]
        )

        change = (
            latest_value -
            previous_value
        )

        if previous_value != 0:

            percentage_change = (
                change /
                abs(previous_value)
            ) * 100

        else:

            percentage_change = 0

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Previous Year",
                int(previous_row["YEAR"])
            )

        with col2:

            st.metric(
                "Present Year",
                int(latest_row["YEAR"])
            )

        with col3:

            st.metric(
                "Percentage Change",
                f"{percentage_change:.2f}%"
            )

        st.divider()

        comparison_df = pd.DataFrame({
            "Year": [
                int(previous_row["YEAR"]),
                int(latest_row["YEAR"])
            ],
            "CO₂": [
                previous_value,
                latest_value
            ]
        })

        st.line_chart(
            comparison_df.set_index("Year")
        )

        if change > 0:

            st.error(
                f"CO₂ emissions increased by "
                f"{abs(change):.3f} units."
            )

        elif change < 0:

            st.success(
                f"CO₂ emissions decreased by "
                f"{abs(change):.3f} units."
            )

        else:

            st.info(
                "CO₂ emissions remained unchanged."
            )

    else:

        st.warning(
            "Not enough historical data available."
        )


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.title("🤖 Machine Learning Model Comparison")

    st.write(
        "Comparison of the machine learning algorithms "
        "used for CO₂ prediction."
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "R² Score Comparison"
    )

    r2_chart = results_df.set_index(
        "Model"
    )["R2 Score"]

    st.bar_chart(
        r2_chart
    )

    st.divider()

    st.subheader(
        "RMSE Comparison"
    )

    rmse_chart = results_df.set_index(
        "Model"
    )["RMSE"]

    st.bar_chart(
        rmse_chart
    )

    st.divider()

    st.success(
        "Primary Model: XGBoost"
    )

    st.info(
        "XGBoost is selected as the primary model "
        "for this project."
    )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "AI-Based Carbon Emission Analysis"
)

st.sidebar.caption(
    "Machine Learning + SHAP"
)
