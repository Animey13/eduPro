import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(page_title="EduPro Forecasting Dashboard", layout="wide")
st.title("Predictive Modeling for Course Demand & Revenue Forecasting")

# Load models and preprocessors
@st.cache_resource
def load_assets():
    enrollment_model = joblib.load('models/best_enrollment_model.pkl')
    revenue_model = joblib.load('models/best_revenue_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    return enrollment_model, revenue_model, preprocessor

enrollment_model, revenue_model, preprocessor = load_assets()

# Load processed data for EDA
@st.cache_data
def load_data():
    course_data = pd.read_csv('data/processed/course_data.csv')
    return course_data

course_data = load_data()

# Helper function for manual feature engineering
def get_features(price, duration, rating, teacher_exp, teacher_rating,
                 category, course_type, level,
                 past_enrollments, past_revenue):

    # Bucket mappings
    if price == 0: pb = 'Free'
    elif price < 100: pb = 'Low'
    elif price < 300: pb = 'Medium'
    else: pb = 'High'

    if duration < 10: db = 'Short'
    elif duration < 30: db = 'Medium'
    else: db = 'Long'

    if rating >= 4.5: rt = 'Excellent'
    elif rating >= 3.5: rt = 'Good'
    elif rating >= 2.5: rt = 'Average'
    else: rt = 'Poor'

    if teacher_exp < 3: tb = 'Junior'
    elif teacher_exp < 7: tb = 'Mid-level'
    else: tb = 'Senior'

    past_avg_rev = past_revenue / max(past_enrollments, 1)

    data = {
        'CourseCategory': [category],
        'CourseType': [course_type],
        'CourseLevel': [level],
        'price_band': [pb],
        'duration_bucket': [db],
        'rating_tier': [rt],
        'instructor_exp_bucket': [tb],
        'CoursePrice': [price],
        'CourseDuration': [duration],
        'CourseRating': [rating],
        'past_enrollment_count': [past_enrollments],
        'past_total_revenue': [past_revenue],
        'past_average_revenue': [past_avg_rev],
        'teacher_avg_experience': [teacher_exp],
        'teacher_avg_rating': [teacher_rating]
    }

    return pd.DataFrame(data)

# Create layout with tabs
tab1, tab2, tab3 = st.tabs(["Course Predictor", "Feature Importance & Analysis", "Category Analysis"])

with tab1:
    st.header("Predict Demand & Revenue for a New Course")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Course Details")
        category = st.selectbox("Category", course_data['CourseCategory'].unique())
        course_type = st.selectbox("Type", course_data['CourseType'].unique())
        level = st.selectbox("Level", course_data['CourseLevel'].unique())
        price = st.number_input("Course Price ($)", min_value=0.0, max_value=2000.0, value=150.0)
        duration = st.number_input("Course Duration (hours)", min_value=1.0, max_value=100.0, value=20.0)
        rating = st.slider("Expected Course Rating", 1.0, 5.0, 4.0)

    with col2:
        st.subheader("Instructor Details")
        teacher_exp = st.slider("Years of Experience", 0, 30, 5)
        teacher_rating = st.slider("Instructor Rating", 1.0, 5.0, 4.5)

    with col3:
        st.subheader("Historical Context (Optional)")
        st.info("For a brand new course, you can leave these at 0, or enter comparable past data.")
        past_enrollments = st.number_input("Past Enrollments", min_value=0, value=0)
        past_revenue = st.number_input("Past Total Revenue", min_value=0.0, value=0.0)

    if st.button("Generate Forecast", type="primary"):
        input_df = get_features(price, duration, rating, teacher_exp, teacher_rating,
                              category, course_type, level,
                              past_enrollments, past_revenue)

        # Preprocess input
        X_processed = preprocessor.transform(input_df)

        # Predict
        predicted_enrollments = enrollment_model.predict(X_processed)[0]
        predicted_revenue = revenue_model.predict(X_processed)[0]

        # Ensure non-negative
        predicted_enrollments = max(0, int(predicted_enrollments))
        predicted_revenue = max(0, predicted_revenue)

        # Display Results
        st.divider()
        st.subheader("Forecast Results")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric("Predicted Enrollments (Next 6 Months)", f"{predicted_enrollments:,}")

        with res_col2:
            st.metric("Predicted Revenue (Next 6 Months)", f"${predicted_revenue:,.2f}")

with tab2:
    st.header("Model Metrics & Feature Importance")

    st.subheader("Lasso Model Performance on Held-Out Test Set")
    metrics_enroll = pd.read_csv('reports/enrollment_metrics.csv')
    metrics_rev = pd.read_csv('reports/revenue_metrics.csv')

    lasso_enroll = metrics_enroll[metrics_enroll['Model'] == 'Lasso Regression']
    lasso_rev = metrics_rev[metrics_rev['Model'] == 'Lasso Regression']

    col_e, col_r = st.columns(2)
    with col_e:
        st.write("**Enrollment Prediction**")
        st.dataframe(lasso_enroll[['MAE', 'RMSE', 'R2']])
    with col_r:
        st.write("**Revenue Prediction**")
        st.dataframe(lasso_rev[['MAE', 'RMSE', 'R2']])

    st.divider()

    st.subheader("Key Drivers of Enrollment Demand")
    fi_df = pd.read_csv('reports/feature_importance.csv')
    top_fi = fi_df[fi_df['Importance'] > 0].head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_fi, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_title("Top 10 Features (Lasso Coefficients)")
    st.pyplot(fig)

    st.info("Insights: Instructor average experience and historical total revenue have the largest positive coefficients, heavily influencing demand. Specific categories (like Digital Marketing) also demonstrate strong positive lift.")

with tab3:
    st.header("Category Analysis (Historical Data)")

    cat_df = course_data.groupby('CourseCategory').agg(
        Total_Revenue=('target_course_revenue', 'sum'),
        Total_Enrollments=('target_enrollment_count', 'sum')
    ).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by Category")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.barplot(data=cat_df.sort_values('Total_Revenue', ascending=False),
                    y='CourseCategory', x='Total_Revenue', ax=ax1, palette='Blues_r')
        st.pyplot(fig1)

    with col2:
        st.subheader("Enrollments by Category")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.barplot(data=cat_df.sort_values('Total_Enrollments', ascending=False),
                    y='CourseCategory', x='Total_Enrollments', ax=ax2, palette='Greens_r')
        st.pyplot(fig2)
