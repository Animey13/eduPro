import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

def engineer_category_features(df_cat, df_courses):
    """
    Engineer features for the category-level model.
    """
    # Aggregate course features to category level
    cat_features = df_courses.groupby('CourseCategory').agg(
        num_courses=('CourseID', 'count'),
        avg_course_price=('CoursePrice', 'mean'),
        avg_course_rating=('CourseRating', 'mean'),
        avg_teacher_rating=('teacher_avg_rating', 'mean'),
        total_past_enrollments=('past_enrollment_count', 'sum'),
        total_past_revenue=('past_total_revenue', 'sum')
    ).reset_index()

    # Merge targets
    df_merged = pd.merge(cat_features, df_cat, on='CourseCategory', how='left')
    return df_merged

def preprocess_category_features(df, is_training=True, preprocessor_path='models/cat_preprocessor.pkl'):
    num_features = ['num_courses', 'avg_course_price', 'avg_course_rating',
                    'avg_teacher_rating', 'total_past_enrollments', 'total_past_revenue']

    X = df[num_features]

    if is_training:
        y_rev = df['target_category_revenue']

        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)

        joblib.dump(scaler, preprocessor_path)
        X_df = pd.DataFrame(X_processed, columns=num_features)

        return X_df, y_rev, num_features
    else:
        scaler = joblib.load(preprocessor_path)
        X_processed = scaler.transform(X)
        X_df = pd.DataFrame(X_processed, columns=num_features)
        return X_df

if __name__ == "__main__":
    df_cat = pd.read_csv('data/processed/category_data.csv')
    df_courses = pd.read_csv('data/processed/course_data.csv')

    df_cat_eng = engineer_category_features(df_cat, df_courses)
    X_cat, y_cat_rev, cat_feat_names = preprocess_category_features(df_cat_eng)

    df_cat_eng.to_csv('data/processed/category_features.csv', index=False)
    X_cat.to_csv('data/processed/X_cat_train.csv', index=False)
    y_cat_rev.to_csv('data/processed/y_cat_rev.csv', index=False)

    print("Category feature engineering complete.")
