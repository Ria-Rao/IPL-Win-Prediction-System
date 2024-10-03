import streamlit as st
import sklearn
import pandas as pd
import pickle


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

pipe = pickle.load(open('pipe1.pkl','rb'))
st.title('IPL Win Predictor')
st.image('back1.png')
# st.image('white.jpg')
teams = sorted(['Gujarat Titans',
 'Mumbai Indians',
 'Chennai Super Kings',
 'Sunrisers Hyderabad',
 'Royal Challengers Bangalore',
 'Lucknow Super Giants',
 'Delhi Capitals',
 'Kolkata Knight Riders',
 'Rajasthan Royals',
 'Kings XI Punjab'])

col1,col2 = st.columns(2)

with col1:
    batting_team =st.selectbox('Select the batting team',teams)
with col2:
    bowling_team = st.selectbox('Select the bowling team',teams)
    
cities = ['Ahmedabad', 'Chennai', 'Mumbai', 'Bengaluru', 'Kolkata', 'Delhi',
       'Dharamsala', 'Hyderabad', 'Lucknow', 'Jaipur', 'Chandigarh',
       'Guwahati', 'Navi Mumbai', 'Pune', 'Dubai', 'Sharjah', 'Abu Dhabi',
       'Visakhapatnam', 'Indore', 'Bangalore', 'Raipur', 'Ranchi',
       'Cuttack', 'Nagpur', 'Johannesburg', 'Centurion', 'Durban',
       'Bloemfontein', 'Port Elizabeth', 'Kimberley', 'East London',
       'Cape Town']

selected_city = st.selectbox('Cities',sorted(cities))

target = st.number_input('Target',min_value=0)

col3,col4,col5 = st.columns(3)
with col3 :
    score =st.number_input('Score',min_value=0)
with col4 :
    wickets =st.number_input('Wickets',min_value=0,max_value=9)
with col5 :
    overs = st.number_input('Overs completed',min_value=0,max_value=20)
    
if st.button('Predict Probability'):
    runs_left = target-score
    balls_left = 120 - overs*6
    wickets = 10-wickets
    crr = score/overs
    rrr = runs_left*6/balls_left
    df =pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'tot_runs_x':[target],'crr':[crr],'rrr':[rrr]})    
    result = pipe.predict_proba(df)
    r_1 = round(result[0][0]*100)
    r_2 = round(result[0][1]*100)
    st.header('Wining Probabilty ')
    st.header(f"{batting_team}  : {r_2} %")
    st.header(f"{bowling_team}  : {r_1} %")
