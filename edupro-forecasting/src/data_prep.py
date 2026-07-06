import pandas as pd
import numpy as np
import os

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    xls = pd.ExcelFile(file_path)
    df_courses = pd.read_excel(xls, 'Courses')
    df_teachers = pd.read_excel(xls, 'Teachers')
    df_transactions = pd.read_excel(xls, 'Transactions')
    print("Data loaded successfully.")
    return df_courses, df_teachers, df_transactions

def prepare_data(df_courses, df_teachers, df_transactions):
    print("Preparing data...")
    # Convert dates
    df_transactions['TransactionDate'] = pd.to_datetime(df_transactions['TransactionDate'])

    # Split transactions into "past" (Jan-Jun) and "future" (Jul-Dec)
    cutoff_date = pd.to_datetime('2025-07-01')
    past_trans = df_transactions[df_transactions['TransactionDate'] < cutoff_date]
    future_trans = df_transactions[df_transactions['TransactionDate'] >= cutoff_date]

    print(f"Past transactions: {len(past_trans)}, Future transactions: {len(future_trans)}")

    # Target Variables (from future transactions)
    # Target 1: Enrollment Count (number of transactions per course)
    # Target 2: Course Revenue (sum of Amount per course)
    course_targets = future_trans.groupby('CourseID').agg(
        target_enrollment_count=('TransactionID', 'count'),
        target_course_revenue=('Amount', 'sum')
    ).reset_index()

    # Historical Features (from past transactions)
    course_history = past_trans.groupby('CourseID').agg(
        past_enrollment_count=('TransactionID', 'count'),
        past_total_revenue=('Amount', 'sum')
    ).reset_index()

    # Calculate derived historical features
    course_history['past_average_revenue'] = course_history['past_total_revenue'] / np.maximum(course_history['past_enrollment_count'], 1)

    # Merge targets and history into courses table
    df_base = pd.merge(df_courses, course_history, on='CourseID', how='left')
    df_base = pd.merge(df_base, course_targets, on='CourseID', how='left')

    # Fill NaN values for courses with no past/future transactions
    fill_cols = ['past_enrollment_count', 'past_total_revenue', 'past_average_revenue',
                 'target_enrollment_count', 'target_course_revenue']
    for col in fill_cols:
        df_base[col] = df_base[col].fillna(0)

    # Teacher features
    # A course might have multiple teachers in transactions, but in the dataset provided, let's see how teachers are mapped.
    # It appears transactions have TeacherID.
    # To get teacher features per course, we aggregate teacher stats based on all transactions (or just past transactions to avoid data leakage).
    # We will use past transactions to aggregate teacher features for a course.

    past_course_teachers = past_trans[['CourseID', 'TeacherID']].drop_duplicates()
    past_course_teachers = pd.merge(past_course_teachers, df_teachers, on='TeacherID', how='left')

    course_teacher_features = past_course_teachers.groupby('CourseID').agg(
        teacher_avg_experience=('YearsOfExperience', 'mean'),
        teacher_avg_rating=('TeacherRating', 'mean')
    ).reset_index()

    df_final = pd.merge(df_base, course_teacher_features, on='CourseID', how='left')

    # Impute missing teacher features (e.g. for courses with no past transactions)
    # Fill with global averages
    global_avg_exp = df_teachers['YearsOfExperience'].mean()
    global_avg_rating = df_teachers['TeacherRating'].mean()

    df_final['teacher_avg_experience'] = df_final['teacher_avg_experience'].fillna(global_avg_exp)
    df_final['teacher_avg_rating'] = df_final['teacher_avg_rating'].fillna(global_avg_rating)

    # Generate Category-level targets
    df_category = df_final.groupby('CourseCategory').agg(
        target_category_revenue=('target_course_revenue', 'sum'),
        target_category_enrollments=('target_enrollment_count', 'sum')
    ).reset_index()

    print(f"Data preparation complete. Output shapes: Courses: {df_final.shape}, Categories: {df_category.shape}")
    return df_final, df_category

if __name__ == "__main__":
    file_path = "data/EduPro Online Platform.xlsx"
    if not os.path.exists(file_path):
        file_path = "../data/EduPro Online Platform.xlsx"

    df_courses, df_teachers, df_transactions = load_data(file_path)
    df_final, df_category = prepare_data(df_courses, df_teachers, df_transactions)

    # Save processed data for downstream tasks
    os.makedirs('data/processed', exist_ok=True)
    df_final.to_csv('data/processed/course_data.csv', index=False)
    df_category.to_csv('data/processed/category_data.csv', index=False)
    print("Processed data saved to data/processed/")
