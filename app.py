import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle

rf_model = joblib.load('random_forest_model.joblib')

teams = ['Kolkata Knight Riders', 'Rajasthan Royals', 'Mumbai Indians',
       'Chennai Super Kings', 'Royal Challengers Bangalore',
       'Sunrisers Hyderabad', 'Delhi Capitals', 'Punjab Kings',
       'Lucknow Super Giants', 'Gujarat Titans', 'Royal Challengers Bengaluru'
       ]

team_mapping = {
    'Royal Challengers Bangalore': 'RCB',
    'Delhi Daredevils': 'DC',
    'Kolkata Knight Riders': 'KKR',
    'Mumbai Indians': 'MI',
    'Rajasthan Royals': 'RR',
    'Chennai Super Kings': 'CSK',
    'Deccan Chargers': 'SRH',
    'Sunrisers Hyderabad': 'SRH',
    'Delhi Capitals': 'DC',
    'Punjab Kings': 'PBKS',
    'Gujarat Titans': 'GT',
    'Lucknow Super Giants': 'LSG',
    'Royal Challengers Bengaluru': 'RCB'
}

bowling_teams = ['RCB', 'MI', 'CSK', 'KKR', 'SRH', 'RR', 'PBKS', 'DC', 'GT', 'LSG']
cities = ['Bangalore', 'Delhi', 'Mumbai', 'Kolkata', 'Hyderabad', 'Chennai',
       'Jaipur', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
       'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
       'Ahmedabad', 'Cuttack', 'Nagpur', 'Visakhapatnam', 'Pune',
       'Raipur', 'Ranchi', 'Abu Dhabi', 'Bengaluru', 'Dubai', 'Sharjah',
       'Navi Mumbai', 'Chandigarh', 'Lucknow', 'Guwahati', 'Dharamsala',
       'Mohali']


st.title("🏏 Cricket Match Outcome Predictor")
st.write("Enter match details to predict the result.")

# User inputs
batting_team = st.selectbox("Batting Team", teams,index=teams.index("Mumbai Indians"))
bowling_team = st.selectbox("Bowling Team", teams,index=teams.index("Sunrisers Hyderabad"))
city = st.selectbox("City", cities,index=cities.index("Hyderabad"))
runs_left = st.number_input("Runs Left", min_value=0, max_value=300, step=1)
balls_left = st.number_input("Balls Left", min_value=0, max_value=120, step=1)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10, step=1)
total_runs_x = st.number_input("Target", min_value=0, max_value=500, step=1)

if balls_left == 120:  # Start of innings
    crr = 0.0
    rr = (runs_left * 6) / balls_left
elif balls_left > 0:
    crr = (total_runs_x - runs_left) / ((120 - balls_left) / 6) if balls_left < 120 else 0.0
    rr = (runs_left * 6) / balls_left
else:
    crr = 0.0
    rr = 0.0


batting_team=team_mapping[batting_team]
bowling_team=team_mapping[bowling_team]


with open("label_encoder_team.pkl", "rb") as f:
    le_team = pickle.load(f)

with open("label_encoder_city.pkl", "rb") as f:
    le_city = pickle.load(f)

le_batting_team = le_team.transform([batting_team])[0]
le_bowling_team = le_team.transform([bowling_team])[0]
city = le_city.transform([city])[0]


input_df = pd.DataFrame([[le_batting_team, le_bowling_team, city, runs_left, balls_left, 
                          wickets_left, total_runs_x, crr, rr]], 
                        columns=['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 
                                 'wickets_left', 'total_runs_x', 'crr', 'rr'])


if st.button("Predict Outcome"):
    prediction = rf_model.predict(input_df)
    if prediction[0]==1:
        st.success(f"Predicted Result: {batting_team} Win")
    else:
        st.success(f"Predicted Result: {bowling_team} Win")
st.write(f'Current Run Rate: {round(crr,2)}')
st.write(f'Required Run Rate: {round(rr,2)}')

team_mapping = {
    'Royal Challengers Bangalore': 'RCB',
    'Delhi Daredevils': 'DC',
    'Kolkata Knight Riders': 'KKR',
    'Mumbai Indians': 'MI',
    'Rajasthan Royals': 'RR',
    'Chennai Super Kings': 'CSK',
    'Deccan Chargers': 'SRH',
    'Sunrisers Hyderabad': 'SRH',
    'Delhi Capitals': 'DC',
    'Punjab Kings': 'PBKS',
    'Gujarat Titans': 'GT',
    'Lucknow Super Giants': 'LSG',
    'Royal Challengers Bengaluru': 'RCB'
}

