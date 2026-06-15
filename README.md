# Cars Price Prediction using Linear Regression

This repository contains a comprehensive data preprocessing and machine learning pipeline to predict car prices based on their technical specifications. The project utilizes a **Linear Regression** model built with `scikit-learn` to establish relationships between vehicle attributes and their corresponding market values.

## 📌 Project Overview
Predicting vehicle prices is a classic regression problem. This project walks through the full data science lifecycle, including:
- **Data Loading & Inspection:** Importing multi-attribute vehicle data with robust encoding handles (`latin1`).
- **Data Cleaning:** Dropping non-predictive columns, identifying and removing duplicate rows, and handling missing (`NaN`) parameters.
- **Feature Engineering & Formatting:** Extracting numerical statistics from text-based specs (e.g., stripping units like `cc`, `hp`, `km/h`, `sec`, and `Nm`).
- **Categorical Encoding:** Label encoding high-cardinality categorical features like `Company Names` and `Engine Types`.
- **Outlier Removal:** Filtering extreme performance metrics to optimize the regression boundaries.
- **Evaluation:** Evaluating prediction performance using R-squared ($R^2$) and Mean Squared Error (MSE) metrics.

## 📊 Dataset Structure
The model processes the following features to determine the **Cars Prices**:
1. **Company Names** (Encoded as Integers)
2. **Engines** (Engine Configuration IDs)
3. **CC/Battery Capacity** (Engine displacement in cc)
4. **HorsePower** (Engine power in hp)
5. **Total Speed** (Top speed in km/h)
6. **Performance (0-100 KM/H)** (Acceleration in seconds)
7. **Fuel Types** (Encoded Fuel category)
8. **Seats** (Number of passenger seats)
9. **Torque** (Engine torque in Nm)

## 🛠️ Tech Stack & Libraries
The notebook is written in **Python 3** and relies on the following key data science libraries:
- **Pandas:** For structured data manipulation, feature cleaning, and index re-alignments.
- **Scikit-Learn (sklearn):** For building the `LinearRegression` model, partitioning training/testing subsets (`train_test_split`), and executing evaluation metrics (`r2_score`, `mean_squared_error`).
- **Seaborn & Matplotlib:** For statistical data visualization and density distribution plotting.
