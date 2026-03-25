import streamlit as st
from streamlit_mic_recorder import mic_recorder

# --- REPLIT-STYLE BLUE THEME ---
st.set_page_config(page_title="AI Smart Interview Analyzer", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0E1117; color: white; }
    
    /* Header Style */
    .main-header { font-size: 35px; font-weight: bold; color: #00D1FF; text-align: center; margin-bottom: 20px; }
    
    /* Box Styling */
    .stTextArea textarea { background-color: #1A1C23; color: white; border: 1px solid #00D1FF; }
    .stButton>button { background-image: linear-gradient(to right, #0072FF, #00C6FF); color: white; border-radius: 20px; border: none; font-weight: bold; }
    
    /* Question Box */
    .q-box { background-color: #1A1C23; padding: 20px; border-radius: 10px; border-left: 5px solid #00D1FF; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 AI Smart Interview Analyzer</p>', unsafe_allow_html=True)

# Question logic
questions = [
    "What is the difference between a list and a tuple in Python?",
    "Explain decorators in Python.",
    "How does memory management work in Python?"
]

if 'q_idx' not in st.session_state:
    st.session_state.q_idx = 0

# UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f'<div class="q-box"><h3>Question {st.session_state.q_idx + 1}</h3><p>{questions[st.session_state.q_idx]}</p></div>', unsafe_allow_html=True)
    st.write("")
    user_text = st.text_area("Type your response here:", height=150)

with col2:
    st.info("🎤 Record your audio response for analysis")
    audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key='recorder')

if st.button("Submit Response"):
    if user_text or audio:
        st.success("Analysis Complete! Great job.")
        if st.session_state.q_idx < len(questions) - 1:
            st.session_state.q_idx += 1
            st.rerun()