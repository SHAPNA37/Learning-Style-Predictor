"""
app.py
Streamlit web app for the Learning Style Predictor.
Loads pre-trained model and predicts student learning style
based on input features, with tailored teaching strategies
and visualizations.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Load model components
model = joblib.load("models/xxgb_learning_style_model.joblib")
scaler = joblib.load("models/scaler_learning_style.joblib")
label_encoder = joblib.load("models/label_encoder_learning_style.joblib")

st.set_page_config(page_title="Learning Style Predictor", layout="wide")
st.title("Student Learning Style Predictor")
st.markdown("Use the sidebar to input student data. Click **Predict** to get results tailored to your teaching strategies.")

# Input Features Sidebar
st.sidebar.header("Enter Student Details")
study_hours = st.sidebar.slider("Study Hours per Day", 0.0, 10.0, 2.0)
sleep_patterns = st.sidebar.slider("Sleep Hours per Day", 0.0, 10.0, 7.0)
screen_time = st.sidebar.slider("Screen Time per Day (hrs)", 0.0, 10.0, 4.0)
motivation = st.sidebar.selectbox("Motivation", ["Low", "Medium", "High"])
class_participation = st.sidebar.selectbox("Class Participation", ["Low", "Medium", "High"])
physical_activity = st.sidebar.selectbox("Physical Activity", ["Low", "Medium", "High"])
edtech_use = st.sidebar.selectbox("Uses Educational Technology?", ["Yes", "No"])
peer_group = st.sidebar.selectbox("Peer Group", ["Negative", "Neutral", "Positive"])
lack_of_interest = st.sidebar.selectbox("Lack of Interest", ["Low", "Medium", "High"])
sports_participation = st.sidebar.selectbox("Sports Participation", ["Low", "Medium", "High"])

if st.sidebar.button("Predict"):
    # Input Preprocessing
    input_dict = {
        'Study_Hours': study_hours,
        'Sleep_Patterns': sleep_patterns,
        'Screen_Time': screen_time,
        'Class_Participation_' + class_participation: 1,
        'Motivation_' + motivation: 1,
        'Physical_Activity_' + physical_activity: 1,
        'Educational_Tech_Use_' + edtech_use: 1,
        'Peer_Group_' + peer_group: 1,
        'Lack_of_Interest_' + lack_of_interest: 1,
        'Sports_Participation_' + sports_participation: 1
    }

    columns_needed = [
        'Study_Hours', 'Sleep_Patterns', 'Screen_Time',
        'Class_Participation_High', 'Class_Participation_Low', 'Class_Participation_Medium',
        'Motivation_High', 'Motivation_Low', 'Motivation_Medium',
        'Physical_Activity_High', 'Physical_Activity_Low', 'Physical_Activity_Medium',
        'Educational_Tech_Use_No', 'Educational_Tech_Use_Yes',
        'Peer_Group_Negative', 'Peer_Group_Neutral', 'Peer_Group_Positive',
        'Lack_of_Interest_High', 'Lack_of_Interest_Low', 'Lack_of_Interest_Medium',
        'Sports_Participation_High', 'Sports_Participation_Low', 'Sports_Participation_Medium'
    ]

    input_data = {col: 0 for col in columns_needed}
    input_data.update(input_dict)
    input_df = pd.DataFrame([input_data])

    numerical_cols = ['Study_Hours', 'Sleep_Patterns', 'Screen_Time']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Prediction
    st.subheader("Predicted Learning Style")
    pred = model.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred])[0]
    st.success(f"**Predicted Learning Style:** {pred_label}")

    # Tailored Recommendations
    st.subheader("Tailored Teaching Strategy")
    if pred_label == "Visual":
        st.info("""
- Use lots of visuals: Whiteboards, diagrams, charts, graphs, pictures, and real objects.  
- Show don't just tell: Demonstrate processes and model problem-solving.  
- Leverage technology: Use videos, animations, and interactive simulations.  
- Employ graphic organizers: Mind maps, concept maps, flowcharts, and timelines.  
- Use color and highlighting: Encourage students to use colors for notes; color-code information.  
- Give visual instructions: Provide written instructions with visual cues or demonstrations.  
- Encourage visual expression: Let students draw, diagram, or create posters to show understanding.  
- Keep it visually clear: Ensure handouts and presentations are organized and easy to see.
        """)
    elif pred_label == "Auditory":
        st.info("""
- Talk it out: Explain concepts verbally, clearly, and concisely.  
- Discuss and debate: Encourage group discussions, debates, and Q&A sessions.  
- Read aloud: Have students read text aloud or listen to audiobooks.  
- Lecture clearly: Deliver well-structured lectures with varying tone and pace.  
- Use sounds: Incorporate music, sound effects, or rhythmic activities when appropriate.  
- Recite and repeat: Encourage students to recite information or repeat it back.  
- Record lessons: Allow students to record lectures or discussions for later listening.
        """)
    elif pred_label == "Kinesthetic":
        st.info("""
- Hands-on activities: Provide opportunities for building, manipulating objects, and experiments.  
- Movement breaks: Allow for movement or short physical activities during lessons.  
- Role-play and dramatize: Use role-playing or acting out concepts.  
- Demonstrate and imitate: Show them how, then let them try it themselves.  
- Tactile materials: Incorporate objects they can touch, feel, or build with.  
- Experiential learning: Organize field trips, practical projects, or simulations.  
- Active participation: Get them involved in classroom setup, moving furniture, or writing on the board.  
- Fidget tools: Allow the use of discreet fidget tools if it aids concentration.
        """)

    # Student Profile Dashboard
    st.markdown("---")
    st.header("Student Profile Dashboard")

    st.subheader("Radar View of Key Traits")
    categories = ['Study_Hours', 'Sleep_Patterns', 'Screen_Time',
                  'Motivation_' + motivation,
                  'Class_Participation_' + class_participation,
                  'Physical_Activity_' + physical_activity,
                  'Lack_of_Interest_' + lack_of_interest,
                  'Sports_Participation_' + sports_participation]
    radar_values = [input_df.get(cat, pd.Series([0])).values[0] for cat in categories]
    labels = [
        'Study_Hours', 'Sleep_Patterns', 'Screen_Time',
        'Motivation', 'Class Participation', 'Physical Activity',
        'Lack of Interest', 'Sports Participation']

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values, theta=labels, fill='toself',
        name='Student Profile', line_color='royalblue'
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, height=400)
    st.plotly_chart(fig_radar)

    st.subheader("Study vs Sleep vs Screen Time Balance")
    bar_df = pd.DataFrame({
        'Metric': ['Study Hours', 'Sleep Hours', 'Screen Time'],
        'Hours': [study_hours, sleep_patterns, screen_time]
    })
    fig2 = plt.figure(figsize=(5, 3))
    sns.barplot(x='Metric', y='Hours', data=bar_df, palette='coolwarm')
    plt.title("Student Time Allocation")
    st.pyplot(fig2)

    st.subheader("Prediction Confidence")
    probs = model.predict_proba(input_df)[0]
    prob_df = pd.DataFrame({
        'Learning Style': label_encoder.inverse_transform(np.arange(len(probs))),
        'Probability': np.round(probs * 100, 2)
    })
    fig3 = plt.figure(figsize=(5, 3))
    sns.barplot(x='Learning Style', y='Probability', data=prob_df, palette='viridis')
    plt.title("Model Confidence in Each Category")
    st.pyplot(fig3)

# Always show Simulated Learning Style Bar Chart
st.markdown("---")
st.header("General Learning Style Distribution (Simulated Data)")
simulated = pd.DataFrame({
    'Learning Style': ['Visual', 'Auditory', 'Kinesthetic'],
    'Count': [34, 34, 32]
})
fig4, ax = plt.subplots()
sns.barplot(x='Learning Style', y='Count', data=simulated, palette='viridis', ax=ax)
plt.title("Simulated Learning Style Distribution")
st.pyplot(fig4)
