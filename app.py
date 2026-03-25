import streamlit as st
from streamlit_mic_recorder import speech_to_text

# Page setup (Replit Blue Theme)
st.set_page_config(page_title="AI Smart Interview", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    .stButton>button { background-color: #007bff; color: white; width: 100%; }
    .score-box { border: 2px solid #007bff; padding: 20px; border-radius: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 AI Smart Interview Analyzer")

# Unga old Replit-la iruntha same questions & answers
questions = [
    {"q": "What is the difference between a list and a tuple?", "a": "immutable"},
    {"q": "What is a Decorator in Python?", "a": "function"},
    {"q": "Explain Python Generators.", "a": "yield"}
]

# Session state to keep track of marks and questions
if 'score' not in st.session_state: st.session_state.score = 0
if 'q_no' not in st.session_state: st.session_state.q_no = 0

if st.session_state.q_no < len(questions):
    curr = questions[st.session_state.q_no]
    
    st.subheader(f"Question {st.session_state.q_no + 1}")
    st.info(curr['q'])

    # VOICE TO TEXT - Neenga pesuna inga type aagum
    st.write("🎤 Click below and speak your answer:")
    text = speech_to_text(language='en', key=f'speech_{st.session_state.q_no}')

    # Manual type box (if voice fails)
    user_input = st.text_area("Your Answer:", value=text if text else "", height=100)

    if st.button("Submit Answer"):
        if user_input:
            # Replit style marking logic
            if curr['a'].lower() in user_input.lower():
                st.session_state.score += 20
                st.success("Correct Answer! +20 Marks")
            else:
                st.error("Incorrect. Moving to next question.")
            
            st.session_state.q_no += 1
            st.rerun()
        else:
            st.warning("Please provide an answer first.")

else:
    # Final Marks Display
    st.markdown(f"""
    <div class="score-box">
        <h2>Interview Over!</h2>
        <h1 style='color: #007bff;'>Total Marks: {st.session_state.score} / 60</h1>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Restart"):
        st.session_state.score = 0
        st.session_state.q_no = 0
        st.rerun()