TransferIQ: Dynamic Football Player Transfer Value Prediction using AI & Multi-Model Approaches

This repository contains the implementation of TransferIQ, a machine learning system designed to predict football player transfer values using multi-source data, advanced ML models, and time-series forecasting.
The project integrates performance metrics, market value trends, sentiment analysis, and injury data to generate a dynamic and data-driven valuation.

🚀 Project Overview

Predicting player transfer values is a complex problem influenced by:

Match performance statistics

Player popularity and public sentiment

Injury history

Contract details

Historical market value trends

Since football player data is not in a continuous time-series format, traditional sequential models like LSTM were unsuitable.
Therefore, the project uses:

XGBoost

LightGBM

Ensemble Models

For the LSTM concept demonstration, a Bitcoin Open–Close price dataset was used because it provides continuous time-series data.

📂 Datasets Used
1. Player Data (Primary Dataset)

Includes:

Player performance stats

Market value trends

Social media sentiment (from Twitter API)

Injury history

Contract details

Issue:
The dataset lacked the sequential structure required for LSTM, hence non-sequential models performed better.

2. Bitcoin Time-Series Dataset (for LSTM)

Used to:

Demonstrate time-series forecasting

Predict open and closing prices

Show LSTM’s temporal learning ability

🛠️ Methodology
1. Data Preprocessing

Handling missing values

Scaling

One-hot encoding

Feature engineering:

Injury risk features

Contract duration

Sentiment scores (VADER/TextBlob)

Performance trend metrics

2. Sentiment Analysis

NLP tools used:

VADER

TextBlob

Features extracted:

Sentiment polarity

Player popularity indicators

🤖 Machine Learning Models
✔ XGBoost

Best for tabular structured data

Handles non-linearity

Robust against noise

✔ LightGBM

Faster gradient boosting

Efficient with large feature sets

✔ Ensemble Model

Combines:

Outputs from XGBoost

Outputs from LightGBM

Additional engineered features

Final model achieved the best stability and accuracy.

✔ LSTM (Proof of Concept)

Implemented on Bitcoin time-series

Demonstrated sequential modeling

Architectures used:

Univariate LSTM

Multivariate LSTM

Encoder–Decoder LSTM
(From PDF architecture on page 3) 

656f0af1-ed66-482b-ae4f-0e587d7…

📊 Model Evaluation

Metrics:

RMSE

MAE

R² Score

Outputs:

Comparison of XGBoost vs LightGBM vs Ensemble

LSTM forecasting curves (Bitcoin dataset)

Player performance visualizations

Market value prediction curves


📌 Key Outcomes

Ensemble model outperformed individual models

LSTM is unsuitable for non-time-series football data

Sentiment + injury + performance features significantly boosted accuracy

System provides dynamic and explainable transfer value predictions

📚 References

StatsBomb Open Data

Transfermarkt Scraping

Twitter API

VADER / TextBlob NLP