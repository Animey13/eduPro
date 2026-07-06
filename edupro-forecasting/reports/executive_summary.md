# Executive Summary: EduPro Predictive Analytics

## Overview
EduPro's course planning has transitioned from intuitive guesses to a data-driven predictive strategy. By analyzing historical transaction patterns, course characteristics, and instructor profiles, we have built a forecasting engine to predict future course enrollments and revenue.

## Key Findings
- **Experience Sells**: The most significant driver of a course's success is the instructor's experience level. Veteran teachers bring massive lift to course enrollments.
- **Success Breeds Success**: A course's historical revenue is highly correlated with its future performance. Early momentum is critical.
- **Revenue is Highly Predictable**: Our models can explain 97% of the variance in future course revenue, allowing for highly accurate budget forecasting. Exact enrollment counts are harder to predict precisely, but the overall revenue generated is very stable.

## Strategic Recommendations
1. **Targeted Instructor Onboarding**: Budget allocations for instructor acquisition should strongly prioritize candidates with 7+ years of experience.
2. **Dynamic Marketing Allocation**: Instead of spreading marketing dollars evenly, double-down on courses that show strong early momentum in their first 6 months.
3. **Strategic Category Focus**: Expand course offerings in naturally high-demand categories such as Digital Marketing, which show an inherent baseline boost in demand.

## Platform Deployment
These models are now live via an interactive Streamlit dashboard. Product managers can input proposed characteristics for a *new* course (Price, Category, Level, Instructor details) and instantly receive a predicted 6-month enrollment and revenue forecast to guide launch decisions.
