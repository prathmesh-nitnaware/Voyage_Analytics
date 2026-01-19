import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import requests
import numpy as np

st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    try:
        f = pd.read_csv("cleaned_flights.csv")
        h = pd.read_csv("cleaned_hotels.csv")
        u = pd.read_csv("cleaned_users.csv")
        return f, h, u
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

flights_df, hotels_df, users_df = load_data()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=80)
st.sidebar.title("Voyage Analytics")
st.sidebar.markdown("AI-Powered Travel Planning")

page = st.sidebar.radio("Navigate", [
    "📊 Business Insights",
    "✈️ Flight Price Predictor",
    "🌍 Smart Trip Planner",  
    "🏨 Hotel Finder"
])

st.sidebar.markdown("---")
st.sidebar.info("System Status")
try:
    check = requests.get("http://localhost:5000/")
    if check.status_code == 200:
        st.sidebar.success("✅ ML Engine Online")
    else:
        st.sidebar.warning("⚠️ ML Engine Unstable")
except:
    st.sidebar.error("❌ ML Engine Offline")


if page == "📊 Business Insights":
    st.title("📊 Market Insights Dashboard")
    st.markdown("### Platform Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Available Flights", f"{len(flights_df):,}")
    c2.metric("Partner Hotels", f"{len(hotels_df):,}")
    c3.metric("Active Users", f"{len(users_df):,}")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✈️ Top Airlines")
        if not flights_df.empty:
            fig, ax = plt.subplots()
            flights_df['agency'].value_counts().plot(kind='bar', ax=ax, color='#4CAF50')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    with col2:
        st.subheader("💰 Ticket Price Range")
        if not flights_df.empty:
            fig, ax = plt.subplots()
            sns.histplot(flights_df['price'], bins=20, kde=True, ax=ax, color='#2196F3')
            st.pyplot(fig)

elif page == "✈️ Flight Price Predictor":
    st.title("✈️ Instant Price Estimator")
    st.markdown("Get real-time price predictions for specific routes using our ML Engine.")
    with st.form("flight_form"):
        c1, c2 = st.columns(2)
        with c1:
            from_city = st.selectbox("Source", flights_df['from'].unique())
            to_city = st.selectbox("Destination", flights_df['to'].unique())
            agency = st.selectbox("Airline", flights_df['agency'].unique())
        with c2:
            flight_type = st.selectbox("Cabin Class", flights_df['flightType'].unique())
            date = st.date_input("Travel Date", datetime.date(2019, 10, 1))
            distance = st.slider("Distance (km)", 100, 2000, 600)
        
        submitted = st.form_submit_button("Predict Price")

    if submitted:
        payload = {
            "from": from_city, "to": to_city, "flightType": flight_type,
            "agency": agency, "time": 1.5, "distance": distance,
            "day": date.day, "month": date.month, "year": date.year
        }
        try:
            response = requests.post("http://localhost:5000/predict", json=payload)
            if response.status_code == 200:
                price = response.json().get('predicted_price', 0)
                st.success(f"### 🏷️ Estimated Fare: R$ {price:,.2f}")
            else:
                st.error("ML Engine Error")
        except:
            st.error("❌ Connection failed. Ensure Docker is running.")

elif page == "🌍 Smart Trip Planner":
    st.title("🌍 Smart Budget Planner")
    st.markdown("Tell us your budget, and **AI will find where you can go.**")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        origin = c1.selectbox("I am leaving from:", flights_df['from'].unique())
        budget = c2.number_input("My Total Budget (R$):", value=1500.0, step=100.0)
        days = c3.slider("Duration (Days):", 1, 7, 3)
    
    if st.button("🚀 Find My Vacation"):
        st.markdown("### 🎯 Recommended Trips")
        
        possible_destinations = [d for d in flights_df['to'].unique() if d != origin]
        valid_trips = []
        
        progress_bar = st.progress(0)
        
        for i, dest in enumerate(possible_destinations):
            payload = {
                "from": origin, "to": dest, "flightType": "firstClass", # Assuming standard comfort
                "agency": "FlyingDrops", # Defaulting to common airline
                "time": 1.5, "distance": 600,
                "day": 10, "month": 10, "year": 2019
            }
            try:
                resp = requests.post("http://localhost:5000/predict", json=payload)
                flight_price = resp.json().get('predicted_price', 0) if resp.status_code == 200 else 9999
            except:
                flight_price = 9999 # Fail safe
            
            city_hotels = hotels_df[hotels_df['place'] == dest]
            if not city_hotels.empty:
                avg_daily_hotel = city_hotels['price'].mean()
                hotel_total = avg_daily_hotel * days
            else:
                hotel_total = 9999
            
            total_trip_cost = flight_price + hotel_total
            
            if total_trip_cost <= budget:
                valid_trips.append({
                    "City": dest,
                    "Flight (Est)": flight_price,
                    "Hotel (Avg)": hotel_total,
                    "Total": total_trip_cost
                })
            
            progress_bar.progress((i + 1) / len(possible_destinations))
            
        progress_bar.empty()

        if valid_trips:
            df_results = pd.DataFrame(valid_trips).sort_values("Total")
            
            for index, trip in df_results.iterrows():
                with st.expander(f"🇧🇷 Trip to {trip['City']} - Total: R$ {trip['Total']:,.2f}"):
                    c1, c2 = st.columns(2)
                    c1.write(f"✈️ **Flight Cost:** R$ {trip['Flight (Est)']:,.2f}")
                    c2.write(f"🏨 **Hotel ({days} days):** R$ {trip['Hotel (Avg)']:,.2f}")
                    st.success("✅ Fits your budget!")
        else:
            st.warning("💸 No trips found under this budget. Try increasing it!")

elif page == "🏨 Hotel Finder":
    st.title("🏨 Hotel Finder")
    c1, c2 = st.columns(2)
    city = c1.selectbox("City", hotels_df['place'].unique() if not hotels_df.empty else [])
    max_p = c2.slider("Max Price/Night", 0, 1000, 400)
    
    res = hotels_df[(hotels_df['place'] == city) & (hotels_df['price'] <= max_p)].sort_values('price')
    
    if not res.empty:
        st.write(f"Found {len(res)} hotels:")
        for _, row in res.head(5).iterrows():
            st.info(f"**{row['name']}** | R$ {row['price']:.2f}/night | {row['days']} Days Package")
    else:
        st.error("No hotels found.")