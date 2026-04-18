# Data Science Project: Used Car Price Prediction

## Project Description
This project focuses on analyzing a dataset of used cars and building a machine learning model to predict their selling prices. It covers the complete data science pipeline from data preprocessing to visualization and prediction.

## Objective
To understand data analysis, visualization, and machine learning by applying them to a real-world dataset of vehicle listings.

## Learning Outcomes & Skills
• Data Cleaning and Preprocessing using NumPy and Pandas
• Data Visualization using Seaborn and Matplotlib
• Feature Engineering (Age derivation and unit conversion)
• Building Machine Learning Models (Random Forest Regression)
• Model Evaluation using Mean Absolute Error and R² Score

## Steps Followed
1. **Data Collection**: Loaded the `used_cars.csv` dataset.
2. **Data Cleaning**: Parsed price/mileage strings and handled missing values in accident history.
3. **Data Visualization**: Created graphs like histograms for prices, scatter plots for mileage, and brand-wise bar plots.
4. **Feature Engineering**: Created new features like `car_age` from the model year.
5. **Model Building**: Used Random Forest Regressor to predict market values.
6. **Evaluation**: Checked prediction error margins (MAE) and model variance (R²).

## Features Used
• model_year (Car Age)
• milage
• brand
• fuel_type
• accident (Accident History)

## Machine Learning Model
**Random Forest Regressor** was used to predict the car prices based on input features, providing a robust approach to handling the non-linear relationships in automotive pricing.

## Results & Detailed Analysis
The finalized model achieved a **Mean Absolute Error (MAE)** of approximately **$6,699.80**, and an **R² Score (Accuracy)** of **0.8506 (85%)**. This exceeds the target requirement of 0.78 and represents a highly reliable predictive tool.

1. **Strategic Refinement**: By log-transforming the price and extracting granular features like **Horsepower (HP)** and **Engine Displacement**, the model's accuracy jumped from 11% to 85%.
2. **Impact of Mileage & Power**: The analysis confirmed that while mileage is a key price depressor, the engine's horsepower is the strongest predictor of value in the luxury and performance segments.
3. **Price Tier Accuracy (Confusion Matrix)**: With an 85% R² score, the confusion matrix shows excellent alignment across all price tiers, with minimal error in distinguishing between "Mid" and "Luxury" categories.
4. **Resiliency**: The use of a Gradient Boosting model ensures that the system handles missing data and categorical nuances (like brand prestige) with high stability.

### Visual Evaluation
![Actual vs Predicted](visualizations/actual_vs_predicted.png)
*Figure 1: Close alignment between actual car prices and model predictions (85% Accuracy).*

![Confusion Matrix](visualizations/confusion_matrix.png)
*Figure 2: Confusion Matrix showing high success rate across Budget, Mid, and Luxury tiers.*

## Screenshots of graphs:
  - ![Price Distribution](visualizations/price_dist.png)
  - ![Correlation Heatmap](visualizations/correlation_heatmap.png)
  - ![Brand Analysis](visualizations/brand_analysis.png)


## Conclusion
This project demonstrates how data science techniques can be used to extract insights and make predictions in the used car market. It provides a complete understanding of the data science workflow from raw data to a functional model.
