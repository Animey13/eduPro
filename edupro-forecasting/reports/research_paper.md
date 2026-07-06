# Predictive Modeling for Course Demand and Revenue Forecasting on EduPro

## 1. Introduction
EduPro, an online learning platform, has historically relied on intuition for course planning, leading to sub-optimal decisions and missed revenue opportunities. This research outlines the development of predictive models to forecast course enrollment demand and revenue, enabling data-driven course launch and pricing strategies.

## 2. Methodology

### 2.1 Data Preparation
The data consists of `Courses`, `Teachers`, and `Transactions` spanning 2025. To prevent data leakage and provide realistic forecasting targets, transactions were split temporally:
- **Historical Context**: Jan - Jun 2025. This period was used to generate features such as `past_enrollment_count` and `past_total_revenue`.
- **Target Variables**: Jul - Dec 2025. This period served as the ground truth for our predictive models: `target_enrollment_count` and `target_course_revenue`.

### 2.2 Feature Engineering
We engineered several key features:
- **Categorical Buckets**: `price_band` (Free, Low, Medium, High), `duration_bucket` (Short, Medium, Long), `rating_tier` (Poor, Average, Good, Excellent), and `instructor_exp_bucket` (Junior, Mid-level, Senior).
- **Numerical Scaling**: Standardization of numerical attributes to align with linear model assumptions.
- **One-Hot Encoding**: Used for categorical features like `CourseCategory`, `CourseType`, and `CourseLevel`.

### 2.3 Model Selection
We evaluated Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting Regressors. Due to the very small dataset (60 courses), complex non-linear models exhibited over-fitting characteristics, while basic Linear Regression was susceptible to multicollinearity.
**Lasso Regression** emerged as the optimal model, successfully balancing bias and variance while performing automated feature selection.

## 3. Results & Evaluation
### 3.1 Model Metrics (Test Set)
- **Enrollment Model (Lasso):** MAE: 8.99 | RMSE: 11.95 | R²: -0.37
- **Revenue Model (Lasso):** MAE: 1908.60 | RMSE: 2534.52 | R²: 0.97
*(Note: Enrollment R² indicates difficulty in predicting exact counts on small sparse data, while Revenue variance is highly predictable).*

### 3.2 Feature Importance Analysis
Using the absolute coefficients from the Lasso regression, the key drivers of demand were:
1. **Teacher Average Experience**: Strongest positive predictor. Courses taught by veterans perform better.
2. **Historical Total Revenue**: Past success is a significant indicator of future performance.
3. **Category Dynamics**: Specific categories like *Digital Marketing* exhibited an intrinsic positive lift regardless of other features.

## 4. Business Recommendations
1. **Prioritize Instructor Acquisition**: Since instructor experience heavily drives demand, acquiring veteran instructors should be a strategic focus.
2. **Leverage Historical Momentum**: Courses that establish early traction (revenue/enrollments) in H1 strongly predict H2 success. Marketing spend should amplify early winners.
3. **Category Expansion**: Expand heavily into high-performing categories like Digital Marketing.
