import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Carbon Emission Analysis",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# LOAD FILES
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_dataset():
    return pd.read_csv("ml_ready_dataset.csv")


@st.cache_data
def load_cleaned_dataset():
    return pd.read_csv("cleaned_dataset.csv")


model = load_model()
df = load_dataset()
cleaned_df = load_cleaned_dataset()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def classify_emission(value):

    if value < 1:
        return "Low"
    elif value < 10:
        return "Moderate"
    elif value < 100:
        return "High"
    else:
        return "Very High"


def get_level_color(level):

    if level == "Low":
        return "green"
    elif level == "Moderate":
        return "orange"
    elif level == "High":
        return "orangered"
    else:
        return "red"


def predict_co2(country, year):

    input_data = pd.DataFrame({
        "YEAR": [year],
        "YEARS_SINCE_1750": [year - 1750],
        "YEAR_SQUARED": [year ** 2],
        "COUNTRY": [country]
    })

    prediction = model.predict(input_data)[0]

    return prediction


def get_shap_explanation(country, year):

    input_data = pd.DataFrame({
        "YEAR": [year],
        "YEARS_SINCE_1750": [year - 1750],
        "YEAR_SQUARED": [year ** 2],
        "COUNTRY": [country]
    })

    # Get model and preprocessing
    xgb_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    # Transform input
    transformed_data = preprocessor.transform(input_data)

    # SHAP
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(transformed_data)

    feature_names = preprocessor.get_feature_names_out()

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0]
    })

    shap_df["Importance"] = shap_df["SHAP Value"].abs()

    # Remove one-hot country features that are zero
    # so that the explanation is easier to understand
    useful_features = []

    for _, row in shap_df.iterrows():

        feature = row["Feature"]

        if "COUNTRY_" in feature:

            if transformed_data[0][list(feature_names).index(feature)] != 0:
                useful_features.append(row)

        else:
            useful_features.append(row)

    if len(useful_features) > 0:
        shap_df = pd.DataFrame(useful_features)

    shap_df = shap_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    return shap_df.head(5), shap_df


def generate_summary(country, year, prediction, level, shap_df):

    positive = shap_df[shap_df["SHAP Value"] > 0]
    negative = shap_df[shap_df["SHAP Value"] < 0]

    summary = (
        f"The model predicts {prediction:.3f} CO₂ emission units "
        f"for {country} in {year}. "
        f"The predicted emission level is {level}."
    )

    if len(positive) > 0:

        feature = positive.iloc[0]["Feature"]

        feature = feature.replace(
            "remainder__", ""
        ).replace(
            "country__", ""
        )

        summary += (
            f" The feature '{feature}' has a positive contribution "
            f"and increases the model's predicted emission value."
        )

    if len(negative) > 0:

        feature = negative.iloc[0]["Feature"]

        feature = feature.replace(
            "remainder__", ""
        ).replace(
            "country__", ""
        )

        summary += (
            f" Other model features have negative contributions "
            f"that reduce the predicted value."
        )

    return summary


def generate_recommendations(level):

    if level == "Low":

        return [
            "Continue monitoring CO₂ emission trends.",
            "Maintain energy-efficient practices.",
            "Increase renewable-energy adoption.",
            "Promote sustainable transportation.",
            "Continue using low-carbon technologies."
        ]

    elif level == "Moderate":

        return [
            "Improve energy efficiency.",
            "Increase renewable-energy usage.",
            "Reduce unnecessary energy consumption.",
            "Promote low-carbon transportation.",
            "Monitor future emission trends regularly."
        ]

    elif level == "High":

        return [
            "Prioritize energy-efficiency improvements.",
            "Increase renewable and low-carbon energy adoption.",
            "Reduce dependence on carbon-intensive energy sources.",
            "Promote energy-efficient transportation.",
            "Increase monitoring of future CO₂ emissions."
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


def get_historical_data(country):

    country_df = df[
        df["COUNTRY"] == country
    ].copy()

    country_df = country_df.sort_values("YEAR")

    return country_df


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌍 Carbon Emission AI")

st.sidebar.markdown(
    """
### Navigation

Use the menu below to explore the system.
"""
)

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🔮 CO₂ Prediction",
        "🔍 SHAP Explanation",
        "💡 Recommendations",
        "📈 Previous vs Present",
        "🤖 Model Comparison"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🌍 AI-Based Carbon Emission Analysis System")

    st.markdown(
        """
        ### Machine Learning Based CO₂ Emission Prediction
        This dashboard uses an XGBoost machine learning model to
        predict CO₂ emissions, classify emission levels, explain
        predictions using SHAP, and provide reduction recommendations.
        """
    )

    st.divider()

    # Dataset statistics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Countries",
            df["COUNTRY"].nunique()
        )

    with col3:
        st.metric(
            "Minimum Year",
            int(df["YEAR"].min())
        )

    with col4:
        st.metric(
            "Maximum Year",
            int(df["YEAR"].max())
        )

    st.divider()

    # Country selection

    countries = sorted(
        df["COUNTRY"].dropna().unique()
    )

    selected_country = st.selectbox(
        "Select Country",
        countries,
        index=countries.index("India")
        if "India" in countries else 0
    )

    latest_year = int(
        df[df["COUNTRY"] == selected_country]["YEAR"].max()
    )

    selected_year = st.number_input(
        "Select Year",
        min_value=int(df["YEAR"].min()),
        max_value=int(df["YEAR"].max()),
        value=latest_year
    )

    if st.button(
        "🔮 Analyze Emission",
        type="primary"
    ):

        prediction = predict_co2(
            selected_country,
            selected_year
        )

        level = classify_emission(prediction)

        st.session_state["country"] = selected_country
        st.session_state["year"] = selected_year
        st.session_state["prediction"] = prediction
        st.session_state["level"] = level

    # Display prediction if available

    if "prediction" in st.session_state:

        prediction = st.session_state["prediction"]
        level = st.session_state["level"]
        country = st.session_state["country"]
        year = st.session_state["year"]

        st.divider()

        st.subheader("Prediction Result")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Country",
                country
            )

        with c2:
            st.metric(
                "Predicted CO₂",
                f"{prediction:.3f}"
            )

        with c3:
            st.metric(
                "Emission Level",
                level
            )

        st.info(
            f"The XGBoost model predicts a **{level}** "
            f"emission level for **{country}** in **{year}**."
        )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    st.title("📊 Dataset Analysis")

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            f"{len(cleaned_df):,}"
        )

    with col2:
        st.metric(
            "Columns",
            len(cleaned_df.columns)
        )

    with col3:
        st.metric(
            "Missing Values",
            int(cleaned_df.isnull().sum().sum())
        )

    st.divider()

    st.subheader("Processed Dataset")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.divider()

    st.subheader("Dataset Columns")

    column_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(df[col].dtype)
            for col in df.columns
        ],
        "Missing Values": [
            int(df[col].isnull().sum())
            for col in df.columns
        ]
    })

    st.dataframe(
        column_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("Country Distribution")

    country_counts = (
        df["COUNTRY"]
        .value_counts()
        .head(15)
    )

    st.bar_chart(country_counts)


# ============================================================
# CO2 PREDICTION
# ============================================================

elif page == "🔮 CO₂ Prediction":

    st.title("🔮 CO₂ Emission Prediction")

    countries = sorted(
        df["COUNTRY"].dropna().unique()
    )

    country = st.selectbox(
        "Select Country",
        countries,
        index=countries.index("India")
        if "India" in countries else 0
    )

    year = st.number_input(
        "Enter Year",
        min_value=int(df["YEAR"].min()),
        max_value=int(df["YEAR"].max()) + 20,
        value=2023
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

        st.session_state["country"] = country
        st.session_state["year"] = year
        st.session_state["prediction"] = prediction
        st.session_state["level"] = level

        st.success("Prediction completed successfully!")

        st.divider()

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

        if level == "Low":
            st.success(
                "🟢 Low emission level"
            )

        elif level == "Moderate":
            st.warning(
                "🟡 Moderate emission level"
            )

        elif level == "High":
            st.warning(
                "🟠 High emission level"
            )

        else:
            st.error(
                "🔴 Very High emission level"
            )


# ============================================================
# SHAP EXPLANATION
# ============================================================

elif page == "🔍 SHAP Explanation":

    st.title("🔍 Why Was This Prediction Made?")

    if "prediction" not in st.session_state:

        st.warning(
            "Please make a prediction first from the CO₂ Prediction page."
        )

    else:

        country = st.session_state["country"]
        year = st.session_state["year"]
        prediction = st.session_state["prediction"]
        level = st.session_state["level"]

        st.write(
            f"### {country} — {year}"
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

        top_features, all_features = get_shap_explanation(
            country,
            year
        )

        st.subheader(
            "Top Contributing Features"
        )

        display_df = top_features.copy()

        display_df["Feature"] = (
            display_df["Feature"]
            .str.replace("remainder__", "", regex=False)
            .str.replace("country__", "", regex=False)
        )

        display_df = display_df[
            ["Feature", "SHAP Value"]
        ]

        st.dataframe(
            display_df,
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "Automatic Explanation"
        )

        summary = generate_summary(
            country,
            year,
            prediction,
            level,
            top_features
        )

        st.info(summary)

        st.divider()

        st.subheader(
            "SHAP Feature Contribution"
        )

        chart_df = top_features.copy()

        chart_df["Feature"] = (
            chart_df["Feature"]
            .str.replace("remainder__", "", regex=False)
            .str.replace("country__", "", regex=False)
        )

        chart_df = chart_df.set_index(
            "Feature"
        )["SHAP Value"]

        st.bar_chart(
            chart_df
        )

        st.caption(
            "Positive SHAP values increase the model prediction; "
            "negative SHAP values decrease it."
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "💡 Recommendations":

    st.title("💡 Carbon Reduction Recommendations")

    if "prediction" not in st.session_state:

        st.warning(
            "Please make a prediction first."
        )

    else:

        country = st.session_state["country"]
        year = st.session_state["year"]
        prediction = st.session_state["prediction"]
        level = st.session_state["level"]

        st.subheader(
            f"Analysis for {country} — {year}"
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

        recommendations = generate_recommendations(
            level
        )

        for recommendation in recommendations:

            st.markdown(
                f"✅ **{recommendation}**"
            )

        st.divider()

        st.info(
            "These recommendations are generated according "
            "to the predicted emission level. They are "
            "sustainability recommendations and should not "
            "be interpreted as causal conclusions from SHAP."
        )


# ============================================================
# PREVIOUS VS PRESENT
# ============================================================

elif page == "📈 Previous vs Present":

    st.title("📈 Previous vs Present CO₂ Emission")

    countries = sorted(
        df["COUNTRY"].dropna().unique()
    )

    country = st.selectbox(
        "Select Country",
        countries,
        index=countries.index("India")
        if "India" in countries else 0
    )

    country_df = get_historical_data(
        country
    )

    if len(country_df) > 0:

        latest_year = int(
            country_df["YEAR"].max()
        )

        previous_year = latest_year - 1

        latest_data = country_df[
            country_df["YEAR"] == latest_year
        ]

        previous_data = country_df[
            country_df["YEAR"] == previous_year
        ]

        if len(latest_data) > 0:

            latest_co2 = latest_data[
                "OBS_VALUE"
            ].iloc[0]

            st.metric(
                "Present CO₂",
                f"{latest_co2:.3f}",
                help=f"Year: {latest_year}"
            )

        if len(previous_data) > 0:

            previous_co2 = previous_data[
                "OBS_VALUE"
            ].iloc[0]

            change = latest_co2 - previous_co2

            if previous_co2 != 0:

                change_percent = (
                    change / previous_co2
                ) * 100

            else:

                change_percent = 0

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Previous CO₂",
                    f"{previous_co2:.3f}"
                )

            with col2:

                st.metric(
                    "Present CO₂",
                    f"{latest_co2:.3f}"
                )

            with col3:

                st.metric(
                    "Change %",
                    f"{change_percent:.2f}%"
                )

            if change > 0:

                st.error(
                    f"📈 CO₂ emissions increased by "
                    f"{abs(change):.3f} units."
                )

            elif change < 0:

                st.success(
                    f"📉 CO₂ emissions decreased by "
                    f"{abs(change):.3f} units."
                )

            else:

                st.info(
                    "CO₂ emissions remained unchanged."
                )

        st.divider()

        st.subheader(
            "Historical CO₂ Emission Trend"
        )

        chart_data = country_df[
            ["YEAR", "OBS_VALUE"]
        ].copy()

        chart_data = chart_data.set_index(
            "YEAR"
        )

        st.line_chart(
            chart_data
        )

        st.caption(
            f"Historical CO₂ emission trend for {country}."
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "🤖 Model Comparison":

    st.title("🤖 Machine Learning Model Comparison")

    st.write(
        "Comparison of the machine learning models used "
        "for CO₂ emission prediction."
    )

    # Try to load saved comparison file

    try:

        comparison_df = pd.read_csv(
            "model_comparison.csv"
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "R² Score Comparison"
        )

        r2_chart = comparison_df[
            ["Model", "R2 Score"]
        ].set_index("Model")

        st.bar_chart(
            r2_chart
        )

        st.divider()

        st.subheader(
            "RMSE Comparison"
        )

        rmse_chart = comparison_df[
            ["Model", "RMSE"]
        ].set_index("Model")

        st.bar_chart(
            rmse_chart
        )

        st.success(
            "Primary model selected for this project: XGBoost"
        )

    except FileNotFoundError:

        st.warning(
            "model_comparison.csv was not found."
        )

        st.info(
            "Upload model_comparison.csv to the project "
            "folder to display model comparison results."
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "AI-Based Analysis of Carbon Emission Using Machine Learning Techniques"
)

st.sidebar.caption(
    "Primary Model: XGBoost"
)
