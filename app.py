import streamlit as st

st.title("AI Smart Interview Analyzer")

# Question list
questions = [
    "What is the difference between a list and a tuple in Python?",
    "Explain decorators in Python.",
    "What are Python generators?"
]

if 'q_idx' not in st.session_state:
    st.session_state.q_idx = 0

# Display Question
st.subheader(f"Question {st.session_state.q_idx + 1}")
st.write(questions[st.session_state.q_idx])

# Answer Section (Only Text/Audio)
user_answer = st.text_area("Your Answer:", placeholder="Type your answer here...")

if st.button("Submit Answer"):
    if user_answer:
        st.success("Answer submitted successfully!")
        # Inga unga analysis logic varum
        if st.session_state.q_idx < len(questions) - 1:
            st.session_state.q_idx += 1
            st.rerun()
    else:
        st.warning("Please type an answer before submitting.")