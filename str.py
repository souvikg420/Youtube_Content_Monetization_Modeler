import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------
# Load Model
# ---------------------
model = joblib.load("youtube_revenue_model.pkl")

# ---------------------
# App Title
# ---------------------
st.title("YouTube Ad Revenue Prediction App")
st.write("Enter video details to predict estimated ad revenue.")

# ---------------------
# User Inputs
# ---------------------
views = st.number_input("Views", min_value=0)
likes = st.number_input("Likes", min_value=0)
comments = st.number_input("Comments", min_value=0)
watch_time = st.number_input("Watch Time (minutes)", min_value=0.0)
video_length = st.number_input("Video Length (minutes)", min_value=0.0)
subscribers = st.number_input("Channel Subscribers", min_value=0)

category = st.selectbox("Category", ["Entertainment", "Gaming", "Education", "Tech", "Music", "Lifestyle"])
device = st.selectbox("Device", ["Mobile", "TV", "Tablet"])
country = st.selectbox("Country", ["IN", "CA", "UK", "DE", "US"])

# ---------------------
# Prepare Input Data
# ---------------------
def prepare_input():
    return pd.DataFrame({
        "views": [views],
        "likes": [likes],
        "comments": [comments],
        "watch_time_minutes": [watch_time],
        "video_length_minutes": [video_length],
        "subscribers": [subscribers],
        "category": [category],
        "device": [device],
        "country": [country]
    })

# ---------------------
# Feature Importance Chart
# ---------------------
st.subheader("Model Feature Importance")

try:
    if hasattr(model, 'named_steps'):
        importances = model.named_steps['model'].coef_
    else:
        importances = model.feature_importances_

    features = list(prepare_input().columns)

    fig2, ax2 = plt.subplots()
    ax2.barh(features, importances)
    ax2.set_xlabel("Importance")
    ax2.set_ylabel("Features")
    st.pyplot(fig2)

except Exception as e:
    st.info("Feature importance not available for this model.")

# ---------------------
# Sample Insights Chart (Simulated)
# ---------------------
st.subheader("Sample Insights: Likes vs Estimated Revenue")

likes_range = np.linspace(0, 5000, 50)
revenue_est = likes_range * 0.03  # dummy trend

fig, ax = plt.subplots()
ax.plot(likes_range, revenue_est)
ax.set_xlabel("Likes")
ax.set_ylabel("Estimated Revenue")
st.pyplot(fig)

# ---------------------
# Engagement vs Revenue Scatter
# ---------------------
st.subheader("Engagement vs Revenue Scatter (Simulated)")

eng = np.random.uniform(0, 0.2, 200)
rev = eng * 400 + np.random.normal(0, 10, 200)

fig3, ax3 = plt.subplots()
ax3.scatter(eng, rev)
ax3.set_xlabel("Engagement Rate")
ax3.set_ylabel("Revenue")
st.pyplot(fig3)

# ---------------------
# Prediction Button
# ---------------------
if st.button("Predict Revenue"):
    input_df = prepare_input()
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Ad Revenue: ${prediction:.2f}")