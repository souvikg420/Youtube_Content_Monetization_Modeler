📊 YouTube Monetization Modeler
Predicting YouTube Ad Revenue Using Machine Learning & Streamlit
🔍 Project Overview

YouTube creators and media companies rely heavily on ad revenue for business planning and content strategy. Accurately predicting ad revenue helps creators optimize content, forecast income, and make data-driven decisions.

This project builds an end-to-end machine learning pipeline to predict YouTube ad revenue for individual videos based on engagement, performance, and contextual features. Multiple regression models are trained and evaluated, and the best-performing model is deployed using a Streamlit web application for interactive predictions.

🎯 Problem Statement

As video monetization becomes increasingly competitive, creators need reliable tools to estimate potential earnings.
The goal of this project is to:

Predict daily YouTube ad revenue (ad_revenue_usd) using historical video performance data.

🗂 Dataset Description

Format: CSV

Rows: ~122,000

Source: Synthetic dataset (created for learning purposes)

Target Variable: ad_revenue_usd

Dataset Features

Each row represents a video’s performance on a specific day.

Feature	Description
video_id	Unique video identifier
date	Date of performance record
views	Number of views
likes	Number of likes
comments	Number of comments
watch_time_minutes	Total watch time
video_length_minutes	Video duration
subscribers	Channel subscriber count
category	Video category
device	Viewer device type
country	Viewer country
ad_revenue_usd	Target variable (Revenue in USD)
🛠 Tech Stack & Tools

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Machine Learning: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting

Deployment: Streamlit

Version Control: Git & GitHub

⚙️ Project Workflow

Data loading and inspection

Handling missing values and duplicates

Exploratory Data Analysis (EDA)

Feature engineering (engagement rate, time-based features)

Encoding categorical variables

Train-test split

Model training and evaluation

Feature importance analysis

Streamlit app deployment

🧪 Feature Engineering

New features created to improve model performance:

Engagement Rate = (likes + comments) / views

Day of Week (from date)

Month (from date)

Engagement per Minute

🤖 Models Trained

The following regression models were trained and compared:

Linear Regression

Ridge Regression

Lasso Regression

Random Forest Regressor

Gradient Boosting Regressor

📊 Evaluation Metrics

R² Score

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

🏆 Model Performance
Model	R²	RMSE	MAE
Linear Regression	~0.95	~13.48	~3.11
Ridge Regression	~0.95	~13.47	~3.11
Lasso Regression	~0.95	~13.47	~3.07
Random Forest	~0.95	~13.81	~3.55
Gradient Boosting	~0.95	~13.52	~3.62

➡ Lasso Regression was selected as the final model.

🔑 Key Business Insights

Engagement (likes & comments) is the strongest driver of revenue

Higher watch time leads to increased monetization

Views directly impact ad impressions and earnings

Subscriber count supports growth but is less impactful than engagement

Category, country, and device have minimal influence compared to engagement metrics

🚀 Streamlit Application

The project includes an interactive Streamlit web app that allows users to:

Enter video performance metrics

Predict estimated YouTube ad revenue

View feature importance

Explore engagement and revenue trends

▶ Run the App Locally
pip install -r requirements.txt
streamlit run app/streamlit_app.py

📁 Project Structure
README.md
regression.ipynb
requirement.txt
str.py
tester.ipynb
youtube_ad_revenue_cleaned.csv
youtube_ad_revenue_dataset.csv

🔮 Future Enhancements

Hyperparameter tuning

Time-series forecasting

SHAP-based explainability

Integration with YouTube Data API

Cloud deployment (Streamlit Cloud)

👤 Author

Souvik Ghosh
Aspiring Data Scientist | Machine Learning Enthusiast
