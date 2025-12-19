import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    page_icon="🎬",
    layout="wide"
)

# =====================================================
# Load Model & Features
# =====================================================
@st.cache_resource(show_spinner="Loading Model...")
def load_assets():
    # Load the model
    model = joblib.load("youtube_revenue_model.pkl")
    
    # Try to get feature names from the model, otherwise use a default list
    try:
        features = model.feature_names_in_.tolist()
    except AttributeError:
        # Fallback: Replace this list with your actual training columns if model.feature_names_in_ fails
        features = [
            "views", "likes", "comments", "watch_time_minutes", 
            "video_length_minutes", "subscribers", "engagement_rate"
        ]
        # Adding dummy columns for categorical placeholders
        categories = ["Entertainment", "Gaming", "Education", "Tech", "Music", "Lifestyle"]
        countries = ["IN", "CA", "UK", "DE", "US"]
        devices = ["Mobile", "TV", "Tablet"]
        
        features += [f"category_{c}" for c in categories]
        features += [f"country_{c}" for c in countries]
        features += [f"device_{d}" for d in devices]
        
    return model, features

model, model_features = load_assets()

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg", width=180)
    st.title("Navigation & Tips")
    st.info("""
    **Revenue Drivers:**
    1. **Watch Time:** Higher retention leads to more ad breaks.
    2. **Region:** US/UK views generally have higher CPM.
    3. **Engagement:** High Like-to-View ratios signal quality to the algorithm.
    """)

# =====================================================
# Main Header
# =====================================================
st.title("🎬 YouTube Ad Revenue Prediction")
st.markdown("Estimate your video's earnings and analyze key performance drivers.")

# =====================================================
# User Input Form
# =====================================================
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📈 Reach")
        views = st.number_input("Views", min_value=0, value=10000, step=1000)
        likes = st.number_input("Likes", min_value=0, value=500)
        comments = st.number_input("Comments", min_value=0, value=50)

    with col2:
        st.subheader("⏱️ Retention")
        watch_time = st.number_input("Total Watch Time (min)", min_value=0.0, value=30000.0)
        video_length = st.number_input("Video Length (min)", min_value=0.0, value=10.0)
        subscribers = st.number_input("Channel Subscribers", min_value=0, value=1000)

    with col3:
        st.subheader("🌍 Context")
        category = st.selectbox("Category", ["Entertainment", "Gaming", "Education", "Tech", "Music", "Lifestyle"])
        device = st.selectbox("Primary Device", ["Mobile", "TV", "Tablet"])
        country = st.selectbox("Top Audience Country", ["US", "IN", "CA", "UK", "DE"])

    submitted = st.form_submit_button("Predict Revenue", use_container_width=True)

# =====================================================
# Prediction Logic
# =====================================================
if submitted:
    try:
        # 1. Feature Engineering
        engagement_rate = (likes + comments) / views if views > 0 else 0
        
        base_data = {
            "views": views,
            "likes": likes,
            "comments": comments,
            "watch_time_minutes": watch_time,
            "video_length_minutes": video_length,
            "subscribers": subscribers,
            "engagement_rate": engagement_rate
        }

        # 2. Alignment with One-Hot Encoding
        df_input = pd.DataFrame([base_data])

        # Initialize all model features to 0
        for col in model_features:
            if col not in df_input.columns:
                df_input[col] = 0
        
        # Set categorical flags to 1
        cat_col = f"category_{category}"
        dev_col = f"device_{device}"
        cnt_col = f"country_{country}"
        
        if cat_col in df_input.columns: df_input[cat_col] = 1
        if dev_col in df_input.columns: df_input[dev_col] = 1
        if cnt_col in df_input.columns: df_input[cnt_col] = 1

        # Reorder columns to match training exactly
        df_input = df_input[model_features]

        # 3. Predict
        prediction = model.predict(df_input)[0]
        # Ensure prediction isn't negative (common in linear models with low inputs)
        prediction = max(0, prediction)

        # 4. Display Results
        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Revenue", f"${prediction:,.2f}")
        m2.metric("Est. RPM", f"${(prediction/views)*1000:,.2f}" if views > 0 else "$0.00")
        m3.metric("Engagement Rate", f"{engagement_rate:.2%}")

        st.balloons()

        # =================================================
        # Visualizations
        # =================================================
        tab1, tab2 = st.tabs(["Feature Impact", "Revenue Projections"])

        with tab1:
            st.subheader("🔍 What influenced this prediction?")
            # Extract coefficients if it's a linear model
            if hasattr(model, 'coef_'):
                importance_df = pd.DataFrame({
                    "Feature": model_features,
                    "Coefficient": model.coef_
                }).sort_values(by="Coefficient", key=abs, ascending=False).head(10)

                fig1, ax1 = plt.subplots()
                colors = ['#FF4B4B' if x > 0 else '#31333F' for x in importance_df["Coefficient"]]
                ax1.barh(importance_df["Feature"], importance_df["Coefficient"], color=colors)
                ax1.set_xlabel("Impact Strength")
                ax1.invert_yaxis()
                st.pyplot(fig1)
            else:
                st.info("Feature importance is available for Lasso/Linear models.")

        with tab2:
            st.subheader("📈 Projected Growth")
            # Simulate revenue scaling with views
            view_scale = np.linspace(views, views*10, 10)
            rev_scale = (prediction / views) * view_scale if views > 0 else view_scale * 0
            
            fig2, ax2 = plt.subplots()
            ax2.plot(view_scale, rev_scale, marker='o', color='#FF0000')
            ax2.set_xlabel("Views")
            ax2.set_ylabel("Revenue ($)")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info("Ensure your model's feature names match the input fields.")