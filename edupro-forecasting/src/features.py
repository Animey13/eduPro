import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def engineer_features(df):
    """
    Apply feature engineering on the course dataset.
    """
    df = df.copy()

    # 1. Price Bands
    # Assuming price ranges. We use pd.qcut or cut. Since some courses are free, we handle zero separately
    def price_band(price):
        if price == 0: return 'Free'
        elif price < 100: return 'Low'
        elif price < 300: return 'Medium'
        else: return 'High'
    df['price_band'] = df['CoursePrice'].apply(price_band)

    # 2. Duration Buckets
    def duration_bucket(duration):
        if duration < 10: return 'Short'
        elif duration < 30: return 'Medium'
        else: return 'Long'
    df['duration_bucket'] = df['CourseDuration'].apply(duration_bucket)

    # 3. Rating Tiers
    def rating_tier(rating):
        if rating >= 4.5: return 'Excellent'
        elif rating >= 3.5: return 'Good'
        elif rating >= 2.5: return 'Average'
        else: return 'Poor'
    df['rating_tier'] = df['CourseRating'].apply(rating_tier)

    # 4. Instructor Experience Buckets
    def exp_bucket(exp):
        if exp < 3: return 'Junior'
        elif exp < 7: return 'Mid-level'
        else: return 'Senior'
    df['instructor_exp_bucket'] = df['teacher_avg_experience'].apply(exp_bucket)

    return df

def preprocess_features(df, is_training=True, preprocessor_path='models/preprocessor.pkl'):
    """
    Encode categorical features and scale numerical features.
    """
    # Define feature columns
    cat_features = ['CourseCategory', 'CourseType', 'CourseLevel',
                    'price_band', 'duration_bucket', 'rating_tier', 'instructor_exp_bucket']

    num_features = ['CoursePrice', 'CourseDuration', 'CourseRating',
                    'past_enrollment_count', 'past_total_revenue', 'past_average_revenue',
                    'teacher_avg_experience', 'teacher_avg_rating']

    target_cols = ['target_enrollment_count', 'target_course_revenue']

    # Extract features
    X = df[cat_features + num_features]

    if is_training:
        y_enroll = df['target_enrollment_count']
        y_rev = df['target_course_revenue']

        # Build Preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
            ])

        X_processed = preprocessor.fit_transform(X)

        # Save preprocessor
        os.makedirs('models', exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)

        # Get feature names for importance analysis
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
        feature_names = num_features + list(cat_feature_names)

        # Create DataFrame
        X_df = pd.DataFrame(X_processed, columns=feature_names)

        return X_df, y_enroll, y_rev, feature_names
    else:
        # Load preprocessor
        preprocessor = joblib.load(preprocessor_path)
        X_processed = preprocessor.transform(X)

        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
        feature_names = num_features + list(cat_feature_names)

        X_df = pd.DataFrame(X_processed, columns=feature_names)
        return X_df

if __name__ == "__main__":
    df = pd.read_csv('data/processed/course_data.csv')
    df_engineered = engineer_features(df)

    X_df, y_enroll, y_rev, feature_names = preprocess_features(df_engineered)

    # Save processed features for training
    df_engineered.to_csv('data/processed/course_features.csv', index=False)
    X_df.to_csv('data/processed/X_train.csv', index=False)
    y_enroll.to_csv('data/processed/y_enroll.csv', index=False)
    y_rev.to_csv('data/processed/y_rev.csv', index=False)
    print(f"Feature engineering complete. Generated {len(feature_names)} features.")
