# Football-Transfer-ML-Pipeline

## About The Project
TransferIQ focuses on building an end-to-end Machine Learning pipeline to predict the market value of football players using historical performance, demographic, public sentiment and transfer-related attributes.

The project covers the complete workflow, starting from data loading to model evaluation, following best practices in data science and machine learning

## Dataset Description
The dataset contains detailed information about football players as:

- Player data (age, position, foot, team name)
- Performance data (goals, assists, passes, total events)
- Injury data (games missed, recovery rate)
- Historical Market Value data
- Public sentiment data

Key Characteristics:

- Structured tabular data
- Contains both numerical and categorical features

## Built Using
- Python
- Google Colab
- NumPy – numerical computations
- Pandas – data manipulation and analysis
- Matplotlib & Seaborn – data visualisation
- Scikit-learn – preprocessing, model training & evaluation
- XGBoost / Random Forest / LightGBM

## Ensemble Learning Overview
Ensemble learning is a method where we use many small models instead of just one. Each of these models may not be very strong on its own, but when we put their results together, we get a better and more accurate answer.

The model produced performs better than the base learners taken alone. Other applications of ensemble learning include selecting the important features, data fusion, etc.

## Getting Started Roadmap

### 1. Loading Dataset in Colab Notebook
- Upload the dataset to Google Colab
- Load the dataset using pandas

### 2. Installing all the required libraries
- Install required dependencies using pip
- Import all general libraries (NumPy, Pandas etc.) at the beginning of the notebook

### 3. Performing Exploratory Data Analysis
- Perform an initial inspection using .head(), .info(), .describe()
- Understand data distribution and structure
- Identify Missing values, Outliers, and feature imbalance
- Analyse target variable (market value)

### 4. Visualising Data
- Plot distributions of numerical features

### 5. Finding correlation
- Compute correlation matrix for numerical features
- Visualize using a heatmap
- Select meaningful predictors for modeling

### 6. Feature engineering
- Create new features from existing columns

### 7. Perform Imputation
- Handle missing values using Mean, Median, Mode
- Ensure no missing values remain before modeling

### 8. Perform scaling
- Normalize numerical features using:
  - StandardScaler
  - MinMaxScaler

### 9. Perform encoding
- Convert categorical variables into numerical form
- Techniques used:
  - One-Hot Encoding
  - Label Encoding

### 10. Train test split
- Split the dataset into:
  - Training set
  - Testing set
- Common split ratio:
  - 80% training
  - 20% testing

### 11. Train model
- Train a machine learning model
- Fit the model using training data
- Tune hyperparameters if required

### 12. Evaluate model
- Evaluate model performance on test data using:
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
- Compare predictions with actual market values and select the best-performing model
