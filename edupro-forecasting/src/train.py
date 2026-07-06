import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return mae, rmse, r2

def train_and_evaluate(X, y, target_name):
    print(f"\nTraining models for: {target_name}")

    # Very small dataset, we should be careful with train/test split.
    # Using 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []
    best_model = None
    best_r2 = -float('inf')
    best_model_name = ""

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)

        # Evaluate
        mae, rmse, r2 = evaluate_model(model, X_test, y_test)

        # Cross-validation on train set
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        cv_r2_mean = cv_scores.mean()

        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_R2': cv_r2_mean
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print(f"Best model: {best_model_name} with R2: {best_r2:.4f}")

    return best_model, best_model_name, results_df

if __name__ == "__main__":
    # Load Course Level Data
    X_course = pd.read_csv('data/processed/X_train.csv')
    y_enroll = pd.read_csv('data/processed/y_enroll.csv')['target_enrollment_count']
    y_rev = pd.read_csv('data/processed/y_rev.csv')['target_course_revenue']

    # Load Category Level Data
    X_cat = pd.read_csv('data/processed/X_cat_train.csv')
    y_cat_rev = pd.read_csv('data/processed/y_cat_rev.csv')['target_category_revenue']

    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # 1. Enrollment Count
    best_model_enroll, name_enroll, res_enroll = train_and_evaluate(X_course, y_enroll, 'Enrollment Count')
    joblib.dump(best_model_enroll, 'models/best_enrollment_model.pkl')
    res_enroll.to_csv('reports/enrollment_metrics.csv', index=False)

    # 2. Course Revenue
    best_model_rev, name_rev, res_rev = train_and_evaluate(X_course, y_rev, 'Course Revenue')
    joblib.dump(best_model_rev, 'models/best_revenue_model.pkl')
    res_rev.to_csv('reports/revenue_metrics.csv', index=False)

    # 3. Category Revenue
    # Since n=12 is very small, we might have weird train/test splits.
    # Let's use it anyway but note it in report.
    best_model_cat, name_cat, res_cat = train_and_evaluate(X_cat, y_cat_rev, 'Category Revenue')
    joblib.dump(best_model_cat, 'models/best_category_revenue_model.pkl')
    res_cat.to_csv('reports/category_revenue_metrics.csv', index=False)

    # Feature Importances for the best Enrollment model
    feature_names = X_course.columns
    if hasattr(best_model_enroll, 'feature_importances_'):
        importances = best_model_enroll.feature_importances_
    elif hasattr(best_model_enroll, 'coef_'):
        importances = np.abs(best_model_enroll.coef_)
    else:
        importances = np.zeros(len(feature_names))

    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False)
    fi_df.to_csv('reports/feature_importance.csv', index=False)

    print("\nTraining complete. Models and metrics saved.")
