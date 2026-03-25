
"""
╔══════════════════════════════════════════════════════════════════════════╗
║        AI-BASED SMART INTERVIEW ANALYZER  ·  College Project Expo        ║
║               Built with Streamlit · Python · DeepFace · NLP             ║
╚══════════════════════════════════════════════════════════════════════════╝

Run:
    streamlit run app.py --server.port 5000

Install deps first:
    pip install streamlit deepface opencv-python-headless sentence-transformers
                SpeechRecognition Pillow numpy tf-keras
"""

# ─── Standard Library ───────────────────────────────────────────────────────
import re
import time
import io
from datetime import datetime

# ─── Third-Party ────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Lazy Imports with error guards ─────────────────────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except Exception:
    DEEPFACE_OK = False

try:
    from sentence_transformers import SentenceTransformer, util
    ST_OK = True
except Exception:
    ST_OK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import speech_recognition as sr
    SR_OK = True
except Exception:
    SR_OK = False

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM CSS
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Smart Interview Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Hero ── */
  .hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d2137 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "";
    position: absolute;
    top: -50px; right: -50px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, #00d4aa33, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4aa, #58a6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.2rem 0;
  }
  .hero-sub { color: #8b949e; font-size: 0.88rem; margin: 0; }

  /* ── Cards ── */
  .card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.4rem 1.7rem;
    margin-bottom: 1rem;
  }
  .card-accent { border-left: 4px solid #00d4aa; }

  /* ── Question block ── */
  .q-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #00d4aa;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  .q-text {
    font-size: 1.1rem;
    font-weight: 500;
    color: #e6edf3;
    line-height: 1.55;
  }
  .topic-badge {
    display: inline-block;
    padding: 3px 12px;
    background: #1f3d4d;
    border: 1px solid #0e6e5e;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #00d4aa;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }

  /* ── Metrics ── */
  .metric-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.1rem;
    text-align: center;
  }
  .metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
  }
  .metric-lbl {
    color: #8b949e;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.3rem;
  }
  .c-green  { color: #3fb950; }
  .c-blue   { color: #58a6ff; }
  .c-orange { color: #f0883e; }
  .c-teal   { color: #00d4aa; }
  .c-red    { color: #f85149; }

  /* ── Emotion badge ── */
  .emotion-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
  }
  .e-happy    { background: #1f4d2e; color: #3fb950; }
  .e-neutral  { background: #1c3a5e; color: #58a6ff; }
  .e-sad      { background: #3d2b1f; color: #f0883e; }
  .e-angry    { background: #4d1f1f; color: #f85149; }
  .e-fear     { background: #4d1f1f; color: #f85149; }
  .e-surprise { background: #2b3d1f; color: #a3c464; }
  .e-disgust  { background: #3d1f3d; color: #c464c4; }
  .e-unknown  { background: #21262d; color: #8b949e; }

  /* ── Suggestion card ── */
  .tip {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
  }
  .tip-icon { font-size: 1.3rem; flex-shrink: 0; }
  .tip-text { font-size: 0.88rem; color: #c9d1d9; line-height: 1.45; }

  /* ── Transcript ── */
  .transcript {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #58a6ff;
    min-height: 54px;
    white-space: pre-wrap;
  }

  /* ── Progress bar override ── */
  .stProgress > div > div { background-color: #00d4aa !important; }

  /* ── Buttons ── */
  .stButton > button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    border-radius: 8px;
    border: 1px solid #30363d;
    background: #21262d;
    color: #e6edf3;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: #00d4aa22;
    border-color: #00d4aa;
    color: #00d4aa;
  }

  /* ── Section heading ── */
  .sec-heading {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }

  /* ── Divider ── */
  hr.thin { border-color: #21262d; margin: 1rem 0; }

  /* ── Report header ── */
  .report-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 0.2rem;
  }
  .report-sub { color: #8b949e; font-size: 0.85rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  QUESTION BANK
# ════════════════════════════════════════════════════════════════════════════
QUESTIONS = [
    {
        "topic": "Python Basics",
        "question": "What is the difference between a list and a tuple in Python?",
        "ideal": (
            "A list is mutable, meaning its elements can be changed after creation, "
            "while a tuple is immutable and cannot be modified. Lists use square brackets "
            "and tuples use parentheses. Tuples are generally faster and used for fixed data, "
            "while lists are used when data needs to change."
        ),
    },
    {
        "topic": "OOP Concepts",
        "question": "Explain inheritance in Object-Oriented Programming with an example.",
        "ideal": (
            "Inheritance allows a child class to acquire the properties and methods of a parent class. "
            "For example, a Dog class can inherit from an Animal class, reusing common attributes like "
            "name and age while adding specific behaviours like bark. It promotes code reusability and "
            "establishes an is-a relationship between classes."
        ),
    },
    {
        "topic": "Databases",
        "question": "What is the difference between SQL and NoSQL databases?",
        "ideal": (
            "SQL databases are relational, structured, and use a fixed schema with tables and rows. "
            "They are ACID compliant and ideal for complex queries. NoSQL databases are non-relational, "
            "schema-less, and can store data as documents, key-value pairs, or graphs. NoSQL is better "
            "for scalability and handling large volumes of unstructured data."
        ),
    },
    {
        "topic": "Data Structures",
        "question": "What is the time complexity of searching in a binary search tree?",
        "ideal": (
            "In a balanced binary search tree the average and best-case time complexity for searching "
            "is O(log n) because each comparison eliminates half the remaining nodes. In the worst case, "
            "if the tree is unbalanced or resembles a linked list, the time complexity degrades to O(n)."
        ),
    },
    {
        "topic": "Machine Learning",
        "question": "What is the role of train-test split in machine learning?",
        "ideal": (
            "Train-test split divides a dataset into two parts: the training set, used to fit the model, "
            "and the test set, used to evaluate its performance on unseen data. This helps detect overfitting "
            "and underfitting. A common split is 80% training and 20% testing. It simulates real-world "
            "deployment and ensures the model generalises well."
        ),
    },
]

FILLER_WORDS = [
    "um", "uh", "ah", "like", "you know", "basically", "literally",
    "actually", "so", "right", "okay", "hmm", "er", "kind of", "sort of",
]

EMOTION_CONFIDENCE = {
    "happy":   95, "neutral": 75, "surprise": 65,
    "sad":     40, "angry":   30, "fear":     25, "disgust": 20,
}

EMOTION_EMOJI = {
    "happy": "😊", "neutral": "😐", "sad": "😢",
    "angry": "😠", "fear": "😨", "surprise": "😲",
    "disgust": "🤢", "unknown": "❓",
}

EMOTION_CSS = {
    "happy": "e-happy", "neutral": "e-neutral", "sad": "e-sad",
    "angry": "e-angry",  "fear": "e-fear",    "surprise": "e-surprise",
    "disgust": "e-disgust", "unknown": "e-unknown",
}

# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = dict(
        phase="interview",          # "interview" | "report"
        q_index=0,                  # current question index
        answers=[],                 # list of answer dicts
        emotions_log=[],            # list of emotion strings
        current_emotion="unknown",  # latest detected emotion
        emotion_scores={},          # raw deepface scores for current capture
        captured_image=None,        # latest annotated PIL image
        transcript="",              # current spoken / typed answer
        st_model=None,              # sentence-transformer model
        model_loaded=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄  Loading NLP model  (first run only)…")
def load_sentence_model():
    if not ST_OK:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        st.warning(f"NLP model error: {exc}")
        return None

# ════════════════════════════════════════════════════════════════════════════
#  ANALYSIS HELPERS
# ════════════════════════════════════════════════════════════════════════════
def compute_similarity(user: str, ideal: str, model) -> float:
    """
    Compute semantic similarity between user answer and ideal answer.
    Priority: sentence-transformers (SBERT) > TF-IDF cosine > keyword overlap.
    """
    if not user.strip():
        return 0.0

    # 1. SBERT — best semantic understanding
    if model:
        try:
            e1 = model.encode(user,  convert_to_tensor=True)
            e2 = model.encode(ideal, convert_to_tensor=True)
            return round(max(0.0, util.cos_sim(e1, e2).item()) * 100, 1)
        except Exception:
            pass

    # 2. TF-IDF cosine similarity — still semantic, much lighter
    if SKLEARN_OK:
        try:
            vec  = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf = vec.fit_transform([user, ideal])
            score = sk_cosine(tfidf[0], tfidf[1])[0][0]
            # TF-IDF cosine tends to be lower; calibrate to 0-100
            calibrated = min(score * 160, 100.0)
            return round(float(calibrated), 1)
        except Exception:
            pass

    # 3. Keyword overlap — basic fallback
    stop = {"a","an","the","is","are","was","were","and","or","but","in","on",
            "at","to","for","of","with","by","from","that","this","it","be"}
    uw = set(user.lower().split()) - stop
    iw = set(ideal.lower().split()) - stop
    return round(len(uw & iw) / max(len(iw), 1) * 100, 1)


def count_fillers(text: str):
    low = text.lower()
    counts = {}
    total = 0
    for f in FILLER_WORDS:
        found = re.findall(r'\b' + re.escape(f) + r'\b', low)
        if found:
            counts[f] = len(found)
            total += len(found)
    return counts, total


def communication_score(text: str, filler_total: int) -> float:
    wc = len(text.split())
    if wc == 0:
        ls = 0
    elif wc < 10:
        ls = 30
    elif wc <= 150:
        ls = 100
    else:
        ls = max(50, 100 - (wc - 150) * 0.3)
    penalty = min(filler_total / max(wc, 1) * 200, 50)
    return round(max(0, min(100, ls - penalty)), 1)


def analyze_emotion(img_array: np.ndarray):
    """Run DeepFace on a numpy RGB image. Returns (dominant, scores_dict)."""
    if not DEEPFACE_OK:
        return "unknown", {}
    try:
        bgr = img_array[:, :, ::-1]          # RGB → BGR for DeepFace
        res = DeepFace.analyze(
            bgr,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if res:
            r = res[0]
            return r.get("dominant_emotion", "unknown").lower(), r.get("emotion", {})
    except Exception:
        pass
    return "unknown", {}


def annotate_image(pil_img: Image.Image, emotion: str, scores: dict) -> Image.Image:
    """Draw emotion overlay on a PIL image and return annotated copy."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    color_map = {
        "happy":   (0, 212, 170), "neutral": (88, 166, 255),
        "sad":     (240, 136, 62), "angry":  (248, 81, 73),
        "fear":    (248, 81, 73),  "surprise":(163, 196, 100),
        "disgust": (196, 100, 196),"unknown": (130, 148, 158),
    }
    color = color_map.get(emotion, (130, 148, 158))

    # Bottom status bar
    bar_h = 44
    draw.rectangle([(0, h - bar_h), (w, h)], fill=(13, 17, 23, 200))
    emoji = EMOTION_EMOJI.get(emotion, "❓")
    label = f"EMOTION: {emotion.upper()}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((12, h - bar_h + 14), label, fill=color, font=font)

    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    draw.text((w - 80, h - bar_h + 14), ts, fill=(130, 148, 158), font=font)

    # Top confidence bar if we have scores
    if scores:
        top_score = max(scores.values()) if scores else 0
        bar_w = int(w * top_score / 100)
        draw.rectangle([(0, 0), (bar_w, 5)], fill=color)

    return img


def confidence_from_emotions(emotions: list) -> float:
    if not emotions:
        return 0.0
    positive = sum(1 for e in emotions if e in ("happy", "neutral", "surprise"))
    return round(positive / len(emotions) * 100, 1)


def dominant_emotion(emotions: list) -> str:
    if not emotions:
        return "unknown"
    from collections import Counter
    return Counter(emotions).most_common(1)[0][0]


def build_suggestions(avg_acc, confidence, avg_comm, avg_filler, dom_emotion):
    tips = []
    if avg_acc < 40:
        tips.append(("📚", "Your technical accuracy is below average. Revise core CS fundamentals before the actual interview."))
    elif avg_acc < 65:
        tips.append(("📖", "Good foundation! Strengthen your answers with precise technical terminology and concrete examples."))
    else:
        tips.append(("🏆", "Excellent technical accuracy! You have a strong command of the subject matter."))

    if dom_emotion in ("sad", "fear", "angry", "disgust"):
        tips.append(("😊", "Your expressions appear stressed. Practise deep breathing and maintain a calm, confident posture."))
    elif dom_emotion == "neutral":
        tips.append(("✨", "Neutral expression is professional. Adding occasional warm smiles will signal more confidence."))
    else:
        tips.append(("💪", "Your positive body language is a strong asset — keep projecting that confidence!"))

    if confidence < 50:
        tips.append(("🧘", "Confidence score is low. Practise mock interviews and record yourself to get comfortable on camera."))

    if avg_filler > 3:
        tips.append(("🎙️", f"You averaged {avg_filler:.1f} filler words per answer. Replace 'um/uh' with a brief pause — silence is powerful."))
    elif avg_filler > 1:
        tips.append(("🗣️", "Slightly reduce words like 'like', 'basically', and 'actually' for a more polished delivery."))
    else:
        tips.append(("🎤", "Outstanding speech clarity! Minimal filler words detected — very professional."))

    if avg_comm < 50:
        tips.append(("📝", "Structure answers with STAR method: Situation → Task → Action → Result for maximum impact."))

    return tips


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 0.5rem 0 1rem 0;'>
          <div style='font-family:Space Mono,monospace; font-size:1.1rem; font-weight:700;
                      background:linear-gradient(90deg,#00d4aa,#58a6ff);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            🎯 Interview AI
          </div>
          <div style='color:#8b949e; font-size:0.72rem; margin-top:4px;'>Expo Prototype v1.0</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 Session Stats")

        total_q = len(QUESTIONS)
        answered = len(st.session_state.answers)
        st.progress(answered / total_q, text=f"Progress: {answered}/{total_q} questions")

        em = st.session_state.current_emotion
        css = EMOTION_CSS.get(em, "e-unknown")
        emoji = EMOTION_EMOJI.get(em, "❓")
        st.markdown(f"""
        <div style='margin: 0.8rem 0;'>
          <div class='sec-heading'>Current Emotion</div>
          <span class='emotion-badge {css}'>{emoji} {em.capitalize()}</span>
        </div>
        """, unsafe_allow_html=True)

        conf = confidence_from_emotions(st.session_state.emotions_log)
        st.metric("Confidence Signal", f"{conf:.0f}%")

        if st.session_state.emotions_log:
            em_counts = {}
            for e in st.session_state.emotions_log:
                em_counts[e] = em_counts.get(e, 0) + 1
            st.markdown("**Emotion History**")
            for e, c in sorted(em_counts.items(), key=lambda x: -x[1]):
                pct = c / len(st.session_state.emotions_log)
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:0.8rem;margin:2px 0;'>"
                    f"<span>{EMOTION_EMOJI.get(e,'?')} {e}</span>"
                    f"<span style='color:#8b949e'>{c}×</span></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("#### ⚙️ Library Status")
        status = [
            ("DeepFace (emotion)",        DEEPFACE_OK),
            ("Sentence-Transformers",     ST_OK),
            ("TF-IDF Similarity",         SKLEARN_OK),
            ("SpeechRecognition",         SR_OK),
        ]
        for name, ok in status:
            icon = "✅" if ok else "⚠️"
            st.markdown(f"{icon} `{name}`")

        if not DEEPFACE_OK:
            st.info("Emotion detection unavailable.\n`pip install deepface tf-keras`")
        if not ST_OK:
            st.info("NLP unavailable.\n`pip install sentence-transformers`")

        st.markdown("---")
        st.caption("College Project Expo · AI Interview Analyzer")

render_sidebar()

# ════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <p class="hero-title">🎯 AI Smart Interview Analyzer</p>
  <p class="hero-sub">
    Real-time facial emotion tracking · Semantic answer evaluation · Communication analytics
  </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  PHASE: REPORT
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "report":
    answers   = st.session_state.answers
    emotions  = st.session_state.emotions_log
    model     = load_sentence_model()

    if not answers:
        st.warning("No answers recorded yet. Complete at least one question first.")
        if st.button("⬅️  Back to Interview"):
            st.session_state.phase = "interview"
            st.rerun()
    else:
        # ── Compute metrics ─────────────────────────────────────────────────
        avg_acc   = round(sum(a["accuracy"]  for a in answers) / len(answers), 1)
        avg_comm  = round(sum(a["comm"]      for a in answers) / len(answers), 1)
        avg_fill  = round(sum(a["filler_total"] for a in answers) / len(answers), 1)
        conf_pct  = confidence_from_emotions(emotions)
        dom_em    = dominant_emotion(emotions)
        overall   = round(avg_acc * 0.45 + avg_comm * 0.30 + conf_pct * 0.25, 1)

        suggestions = build_suggestions(avg_acc, conf_pct, avg_comm, avg_fill, dom_em)

        # ── Header ──────────────────────────────────────────────────────────
        st.markdown("""
        <div class='report-header'>📈 Interview Performance Report</div>
        <div class='report-sub'>
          Generated on """ + datetime.now().strftime("%d %B %Y, %H:%M") + """
        </div>""", unsafe_allow_html=True)

        # ── Overall score big banner ─────────────────────────────────────────
        if overall >= 70:
            score_color, grade, grade_txt = "c-green", "A", "Excellent"
        elif overall >= 50:
            score_color, grade, grade_txt = "c-orange", "B", "Good"
        else:
            score_color, grade, grade_txt = "c-red", "C", "Needs Work"

        st.markdown(f"""
        <div class='card' style='text-align:center; padding: 2rem;'>
          <div style='font-family:Space Mono,monospace; font-size:0.7rem; color:#8b949e;
                      letter-spacing:2px; text-transform:uppercase; margin-bottom:0.5rem;'>
            Overall Performance Score
          </div>
          <div class='metric-val {score_color}' style='font-size:4rem;'>{overall:.0f}%</div>
          <div style='font-size:1.1rem; margin-top:0.4rem; color:#8b949e;'>
            Grade: <strong style='color:#e6edf3;'>{grade}</strong> · {grade_txt}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Four metric cards ────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            col = "c-green" if avg_acc >= 70 else ("c-orange" if avg_acc >= 45 else "c-red")
            st.markdown(f"""<div class='metric-box'>
              <div class='metric-val {col}'>{avg_acc:.0f}%</div>
              <div class='metric-lbl'>Technical Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            col = "c-green" if conf_pct >= 70 else ("c-orange" if conf_pct >= 40 else "c-red")
            st.markdown(f"""<div class='metric-box'>
              <div class='metric-val {col}'>{conf_pct:.0f}%</div>
              <div class='metric-lbl'>Confidence Score</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            col = "c-green" if avg_comm >= 70 else ("c-orange" if avg_comm >= 45 else "c-red")
            st.markdown(f"""<div class='metric-box'>
              <div class='metric-val {col}'>{avg_comm:.0f}%</div>
              <div class='metric-lbl'>Communication</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            emoji = EMOTION_EMOJI.get(dom_em, "❓")
            st.markdown(f"""<div class='metric-box'>
              <div class='metric-val' style='font-size:2rem;'>{emoji}</div>
              <div style='font-family:Space Mono,monospace; font-size:0.9rem; font-weight:700;
                          color:#e6edf3; margin-top:4px;'>{dom_em.capitalize()}</div>
              <div class='metric-lbl'>Dominant Mood</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='thin'>", unsafe_allow_html=True)

        # ── Per-question breakdown ───────────────────────────────────────────
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("<div class='sec-heading'>📋 Per-Question Breakdown</div>", unsafe_allow_html=True)
            for i, ans in enumerate(answers):
                q = QUESTIONS[ans["q_index"]]
                a_col = "c-green" if ans["accuracy"] >= 70 else ("c-orange" if ans["accuracy"] >= 45 else "c-red")
                c_col = "c-green" if ans["comm"]     >= 70 else ("c-orange" if ans["comm"]     >= 45 else "c-red")
                st.markdown(f"""
                <div class='card card-accent' style='margin-bottom:0.7rem;'>
                  <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                      <span class='topic-badge'>{q['topic']}</span>
                      <div style='font-size:0.88rem; color:#c9d1d9; margin-top:2px;
                                  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:340px;'>
                        Q{i+1}: {q['question'][:80]}{'…' if len(q['question'])>80 else ''}
                      </div>
                    </div>
                    <div style='display:flex; gap:10px; flex-shrink:0;'>
                      <div style='text-align:center;'>
                        <div class='metric-val {a_col}' style='font-size:1.4rem;'>{ans["accuracy"]:.0f}</div>
                        <div class='metric-lbl'>acc</div>
                      </div>
                      <div style='text-align:center;'>
                        <div class='metric-val {c_col}' style='font-size:1.4rem;'>{ans["comm"]:.0f}</div>
                        <div class='metric-lbl'>comm</div>
                      </div>
                      <div style='text-align:center;'>
                        <div class='metric-val' style='font-size:1.4rem; color:#f0883e;'>{ans["filler_total"]}</div>
                        <div class='metric-lbl'>fillers</div>
                      </div>
                    </div>
                  </div>
                  {f'<div style="margin-top:0.6rem; font-family:Space Mono,monospace; font-size:0.78rem; color:#58a6ff;">&quot;{ans["transcript"][:120]}{"…" if len(ans["transcript"])>120 else ""}&quot;</div>' if ans["transcript"] else ''}
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='sec-heading'>💡 AI Feedback & Suggestions</div>", unsafe_allow_html=True)
            for icon, text in suggestions:
                st.markdown(f"""
                <div class='tip'>
                  <div class='tip-icon'>{icon}</div>
                  <div class='tip-text'>{text}</div>
                </div>
                """, unsafe_allow_html=True)

            # Filler words detail
            if any(ans["fillers"] for ans in answers):
                st.markdown("<div class='sec-heading' style='margin-top:1rem;'>🔤 Top Filler Words Detected</div>", unsafe_allow_html=True)
                combined: dict = {}
                for ans in answers:
                    for w, c in ans["fillers"].items():
                        combined[w] = combined.get(w, 0) + c
                top = sorted(combined.items(), key=lambda x: -x[1])[:6]
                cols = st.columns(3)
                for idx, (w, c) in enumerate(top):
                    with cols[idx % 3]:
                        st.markdown(f"""
                        <div class='metric-box'>
                          <div class='metric-val c-orange' style='font-size:1.6rem;'>{c}</div>
                          <div class='metric-lbl'>"{w}"</div>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("<hr class='thin'>", unsafe_allow_html=True)
        if st.button("🔄  Start New Interview Session", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
#  PHASE: INTERVIEW
# ════════════════════════════════════════════════════════════════════════════
else:  # phase == "interview"
    model = load_sentence_model()
    q_idx = st.session_state.q_index
    total = len(QUESTIONS)

    # ── Progress bar ─────────────────────────────────────────────────────────
    st.progress(q_idx / total, text=f"Question {q_idx + 1} of {total}")

    # ── Two-column layout ────────────────────────────────────────────────────
    cam_col, interview_col = st.columns([1, 1], gap="large")

    # ────────────────────────────────────────────────────────────────────────
    # LEFT COLUMN — Live Camera & Emotion Detection
    # ────────────────────────────────────────────────────────────────────────
    with cam_col:
        st.markdown("<div class='sec-heading'>📸 Live Emotion Detection</div>", unsafe_allow_html=True)

        # Mode toggle
        capture_mode = st.radio(
            "Camera Mode",
            ["📷 Snapshot (Recommended)", "⬆️ Upload Photo"],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.caption("Tip: Capture a snapshot before or during your answer so the AI can track your confidence level.")

        raw_img_data = None

        if "Snapshot" in capture_mode:
            camera_snap = st.camera_input(
                "Point camera at your face and click the shutter button",
                label_visibility="collapsed",
                key=f"cam_{q_idx}",
            )
            if camera_snap:
                raw_img_data = camera_snap

        else:
            uploaded = st.file_uploader(
                "Upload a photo of yourself",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key=f"up_{q_idx}",
            )
            if uploaded:
                raw_img_data = uploaded

        # ── Analyse and display ─────────────────────────────────────────────
        if raw_img_data is not None:
            pil_img   = Image.open(raw_img_data).convert("RGB")
            img_array = np.array(pil_img)

            with st.spinner("🔍 Analyzing facial emotion…"):
                emotion, scores = analyze_emotion(img_array)

            # Update session state
            st.session_state.current_emotion = emotion
            st.session_state.emotions_log.append(emotion)

            # Annotate and display
            annotated = annotate_image(pil_img, emotion, scores)
            st.image(annotated, caption="Live feed — emotion annotated", use_container_width=True)

            # Emotion badge
            css   = EMOTION_CSS.get(emotion, "e-unknown")
            emoji = EMOTION_EMOJI.get(emotion, "❓")
            st.markdown(f"""
            <div style='margin-top:0.5rem;'>
              <span class='emotion-badge {css}' style='font-size:1rem; padding:8px 18px;'>
                {emoji}  Detected: {emotion.upper()}
              </span>
            </div>
            """, unsafe_allow_html=True)

            # Mini score bars
            if scores:
                st.markdown("**Emotion Confidence Breakdown:**")
                top_emotions = sorted(scores.items(), key=lambda x: -x[1])[:4]
                for e_name, e_score in top_emotions:
                    bar_col = "#00d4aa" if e_name == emotion else "#30363d"
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0;"
                        f"font-size:0.78rem;'>"
                        f"<span style='width:68px;color:#8b949e;'>{e_name}</span>"
                        f"<div style='flex:1;background:#21262d;border-radius:4px;height:8px;'>"
                        f"<div style='width:{e_score:.0f}%;background:{bar_col};height:8px;"
                        f"border-radius:4px;'></div></div>"
                        f"<span style='width:36px;text-align:right;color:#8b949e;'>{e_score:.0f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        else:
            # Placeholder
            st.markdown("""
            <div class='card' style='text-align:center; min-height:280px; display:flex;
                 flex-direction:column; align-items:center; justify-content:center; gap:12px;'>
              <div style='font-size:3rem;'>📷</div>
              <div style='color:#8b949e; font-size:0.9rem;'>
                Capture or upload a photo of your face<br>so the AI can analyse your expression
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Running confidence meter
        if st.session_state.emotions_log:
            conf_running = confidence_from_emotions(st.session_state.emotions_log)
            st.markdown("<hr class='thin'>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; margin-bottom:4px;
                        font-size:0.78rem; color:#8b949e;'>
              <span>Session Confidence</span>
              <span style='color:#00d4aa; font-weight:600;'>{conf_running:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(conf_running / 100)

    # ────────────────────────────────────────────────────────────────────────
    # RIGHT COLUMN — Question & Answer
    # ────────────────────────────────────────────────────────────────────────
    with interview_col:
        q = QUESTIONS[q_idx]

        # Question display
        st.markdown(f"""
        <div class='card card-accent'>
          <span class='topic-badge'>{q['topic']}</span>
          <div class='q-label'>Question {q_idx + 1} / {total}</div>
          <div class='q-text'>{q['question']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Answer input ─────────────────────────────────────────────────────
        st.markdown("<div class='sec-heading'>✍️ Your Answer</div>", unsafe_allow_html=True)
        input_mode = st.radio(
            "Input mode",
            ["🖊️ Type Answer", "🎤 Use Microphone", "📁 Upload Audio"],
            horizontal=True,
            label_visibility="collapsed",
        )

        transcript = ""

        if "Type" in input_mode:
            transcript = st.text_area(
                "Type your answer here",
                value=st.session_state.transcript,
                height=180,
                placeholder="Write your answer here. Aim for 40–120 words for a complete, well-structured response…",
                label_visibility="collapsed",
                key=f"ta_{q_idx}",
            )
            st.session_state.transcript = transcript

        elif "Microphone" in input_mode:
            st.info(
                "⚠️  Microphone recording runs server-side. "
                "For a local install, click **Start Recording** below. "
                "In a hosted environment, use the Upload Audio option instead."
            )
            if st.button("🎙️  Start Recording (8 seconds)", use_container_width=True):
                if not SR_OK:
                    st.error("SpeechRecognition not installed. Run `pip install SpeechRecognition pyaudio`.")
                else:
                    with st.spinner("🎤 Listening for 8 seconds… Speak clearly!"):
                        recognizer = sr.Recognizer()
                        try:
                            with sr.Microphone() as source:
                                recognizer.adjust_for_ambient_noise(source, duration=0.8)
                                audio = recognizer.listen(source, timeout=10, phrase_time_limit=60)
                            transcript = recognizer.recognize_google(audio)
                            st.session_state.transcript = transcript
                            st.success(f"✅ Transcribed: {transcript}")
                        except sr.WaitTimeoutError:
                            st.warning("⏳ No speech detected — please try again.")
                        except sr.UnknownValueError:
                            st.warning("🔇 Could not understand. Please speak more clearly.")
                        except sr.RequestError as e:
                            st.error(f"API error: {e}")
                        except OSError:
                            st.error("🎤 Microphone not found. Check device permissions or use Upload Audio.")
                        except Exception as e:
                            st.error(f"Error: {e}")
            if st.session_state.transcript:
                st.markdown(f"<div class='transcript'>{st.session_state.transcript}</div>", unsafe_allow_html=True)
            transcript = st.session_state.transcript

        else:  # Upload Audio
            audio_file = st.file_uploader(
                "Upload an audio file (WAV, AIFF, FLAC)",
                type=["wav", "aiff", "flac"],
                label_visibility="collapsed",
                key=f"audio_{q_idx}",
            )
            if audio_file:
                if not SR_OK:
                    st.error("SpeechRecognition not installed.")
                else:
                    with st.spinner("🔍 Transcribing audio…"):
                        recognizer = sr.Recognizer()
                        try:
                            audio_bytes = audio_file.read()
                            audio_io    = io.BytesIO(audio_bytes)
                            with sr.AudioFile(audio_io) as source:
                                audio_data = recognizer.record(source)
                            transcript = recognizer.recognize_google(audio_data)
                            st.session_state.transcript = transcript
                            st.success("✅ Transcription complete!")
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")
                if st.session_state.transcript:
                    st.markdown(f"<div class='transcript'>{st.session_state.transcript}</div>", unsafe_allow_html=True)
                transcript = st.session_state.transcript

        # Live word count & filler preview
        if transcript.strip():
            wc = len(transcript.split())
            _, ft = count_fillers(transcript)
            c_left, c_right = st.columns(2)
            c_left.metric("Word Count", wc,
                          delta="Good length" if 40 <= wc <= 120 else ("Too short" if wc < 40 else "Too long"),
                          delta_color="normal" if 40 <= wc <= 120 else "inverse")
            c_right.metric("Filler Words", ft,
                           delta="Clean!" if ft == 0 else f"{ft} detected",
                           delta_color="normal" if ft == 0 else "inverse")

        # ── Submit answer ─────────────────────────────────────────────────────
        st.markdown("<hr class='thin'>", unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns([2, 1])

        with btn_col1:
            if st.button("✅  Submit & Analyse Answer", type="primary", use_container_width=True, key=f"sub_{q_idx}"):
                if not transcript.strip():
                    st.error("Please provide an answer before submitting.")
                else:
                    with st.spinner("🧠 Computing semantic similarity…"):
                        acc  = compute_similarity(transcript, q["ideal"], model)
                        f_d, f_t = count_fillers(transcript)
                        comm = communication_score(transcript, f_t)

                    st.session_state.answers.append({
                        "q_index":      q_idx,
                        "transcript":   transcript,
                        "accuracy":     acc,
                        "comm":         comm,
                        "fillers":      f_d,
                        "filler_total": f_t,
                    })
                    st.session_state.transcript = ""

                    # Show instant feedback
                    a_col = "#3fb950" if acc >= 70 else ("#f0883e" if acc >= 45 else "#f85149")
                    c_col = "#3fb950" if comm >= 70 else ("#f0883e" if comm >= 45 else "#f85149")
                    st.markdown(f"""
                    <div class='card' style='margin-top:0.5rem;'>
                      <div class='sec-heading'>⚡ Instant Feedback</div>
                      <div style='display:flex; gap:1.5rem; margin-bottom:0.8rem;'>
                        <div>
                          <div class='metric-val' style='color:{a_col}; font-size:2rem;'>{acc:.0f}%</div>
                          <div class='metric-lbl'>Accuracy</div>
                        </div>
                        <div>
                          <div class='metric-val' style='color:{c_col}; font-size:2rem;'>{comm:.0f}%</div>
                          <div class='metric-lbl'>Communication</div>
                        </div>
                        <div>
                          <div class='metric-val' style='color:#f0883e; font-size:2rem;'>{f_t}</div>
                          <div class='metric-lbl'>Fillers</div>
                        </div>
                      </div>
                      <div style='font-size:0.85rem; color:#8b949e; border-top:1px solid #21262d;
                                  padding-top:0.7rem; margin-top:0.2rem;'>
                        {"✅ Strong answer — great semantic overlap!" if acc >= 70 else
                         "📖 Partial coverage — try to include more specific technical detail." if acc >= 45 else
                         "📚 Low accuracy — review this topic before the interview."}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Advance or finish
                    if q_idx < total - 1:
                        st.session_state.q_index += 1
                        time.sleep(1.8)
                        st.rerun()
                    else:
                        time.sleep(1.5)
                        st.session_state.phase = "report"
                        st.rerun()

        with btn_col2:
            if st.button("⏭️  Skip Question", use_container_width=True, key=f"skip_{q_idx}"):
                st.session_state.transcript = ""
                if q_idx < total - 1:
                    st.session_state.q_index += 1
                    st.rerun()
                else:
                    st.session_state.phase = "report"
                    st.rerun()

        # Ideal answer peek (collapsible)
        with st.expander("💡 View Model Answer (for reference only)"):
            st.markdown(f"""
            <div style='font-size:0.88rem; color:#8b949e; line-height:1.6;
                        font-style:italic; border-left:3px solid #00d4aa; padding-left:1rem;'>
              {q['ideal']}
            </div>
            """, unsafe_allow_html=True)

    # ── Generate Report button ───────────────────────────────────────────────
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    answered_count = len(st.session_state.answers)

    gcol1, gcol2, gcol3 = st.columns([1, 2, 1])
    with gcol2:
        if st.button(
            f"📊  Generate Full Report  ({answered_count}/{total} answered)",
            type="primary",
            use_container_width=True,
            disabled=(answered_count == 0),
        ):
            st.session_state.phase = "report"
            st.rerun()

