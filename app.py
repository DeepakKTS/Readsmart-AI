import streamlit as st
import pdfplumber
import torch
import random
import re
from transformers import pipeline, T5Tokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from io import BytesIO

# ------------------- Setup -------------------
# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qg_tokenizer = T5Tokenizer.from_pretrained("iarfmoose/t5-base-question-generator", use_fast=False)
qg_model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator").to(device)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# ------------------- Streamlit Config -------------------
st.set_page_config(page_title="📘 ReadSmart AI", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Global Styles */
html, body, [class*="css"] {
    background: linear-gradient(135deg, #050a14 0%, #0f1a2e 50%, #1a2332 100%);
    color: #FFFFFF;
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #050a14 0%, #0f1a2e 50%, #1a2332 100%);
}

/* Hero Text */
.hero-text {
    font-size: 3.5rem;
    margin: 2rem 0;
    text-align: center;
    color: #00FFE0;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(0, 255, 224, 0.5);
}

/* Center Button Container */
.center-button {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 30px auto;
    width: 100%;
}

/* Streamlit Button Styling with Integrated Rainbow Glow */
.stButton {
    display: flex;
    justify-content: center;
    width: 100%;
}

.stButton > button {
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(20px) !important;
    border: 3px solid white !important;
    border-radius: 50px !important;
    color: white !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
    padding: 1rem 2.5rem !important;
    transition: all 0.3s ease !important;
    width: auto !important;
    height: auto !important;
    min-width: 300px !important;
    outline: none !important;
    margin: 0 !important;
    position: relative !important;
    box-shadow: 
        0 0 0 4px conic-gradient(from 0deg, #00ffe0, #8a2be2, #ff1493, #ff8c00, #00ffe0),
        0 0 40px rgba(0, 255, 224, 0.8),
        0 0 80px rgba(0, 255, 224, 0.4),
        0 0 120px rgba(0, 255, 224, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    animation: rainbow-glow 3s linear infinite !important;
}

@keyframes rainbow-glow {
    0% {
        box-shadow: 
            0 0 0 4px #00ffe0,
            0 0 40px rgba(0, 255, 224, 0.8),
            0 0 80px rgba(0, 255, 224, 0.4),
            0 0 120px rgba(0, 255, 224, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    25% {
        box-shadow: 
            0 0 0 4px #8a2be2,
            0 0 40px rgba(138, 43, 226, 0.8),
            0 0 80px rgba(138, 43, 226, 0.4),
            0 0 120px rgba(138, 43, 226, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    50% {
        box-shadow: 
            0 0 0 4px #ff1493,
            0 0 40px rgba(255, 20, 147, 0.8),
            0 0 80px rgba(255, 20, 147, 0.4),
            0 0 120px rgba(255, 20, 147, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    75% {
        box-shadow: 
            0 0 0 4px #ff8c00,
            0 0 40px rgba(255, 140, 0, 0.8),
            0 0 80px rgba(255, 140, 0, 0.4),
            0 0 120px rgba(255, 140, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    100% {
        box-shadow: 
            0 0 0 4px #00ffe0,
            0 0 40px rgba(0, 255, 224, 0.8),
            0 0 80px rgba(0, 255, 224, 0.4),
            0 0 120px rgba(0, 255, 224, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
}

.stButton > button:hover {
    transform: scale(1.05) !important;
    background: rgba(30, 41, 59, 0.8) !important;
    border: 3px solid white !important;
    outline: none !important;
}

.stButton > button:focus {
    outline: none !important;
    border: 3px solid white !important;
}

.stButton > button:active {
    outline: none !important;
    border: 3px solid white !important;
    transform: scale(0.98) !important;
}

.stButton > button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    outline: none !important;
    border: 3px solid white !important;
}

/* Professional Minimalist Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: transparent !important;
    justify-content: center !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding-bottom: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: rgba(255, 255, 255, 0.6) !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 500 !important;
    padding: 1rem 2rem !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: rgba(255, 255, 255, 0.8) !important;
    background: rgba(255, 255, 255, 0.02) !important;
}

.stTabs [aria-selected="true"] {
    color: #00FFE0 !important;
    background: transparent !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"]::after {
    content: '' !important;
    position: absolute !important;
    bottom: -1px !important;
    left: 0 !important;
    right: 0 !important;
    height: 2px !important;
    background: linear-gradient(90deg, #00FFE0, #8a2be2) !important;
    border-radius: 1px !important;
}

/* File Uploader Styling */
.stFileUploader > div > div > div {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 2px solid #64748b !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
}

.stFileUploader > div > div > div:hover {
    border-color: rgba(0, 255, 224, 0.5) !important;
}

.stFileUploader label {
    color: #FFFFFF !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 500 !important;
}

/* Success/Info/Warning Messages */
.stSuccess {
    background: rgba(30, 41, 59, 0.8) !important;
    border-radius: 12px !important;
    border-left: 4px solid #00FFE0 !important;
    color: #FFFFFF !important;
}

.stInfo {
    background: rgba(30, 41, 59, 0.8) !important;
    border-radius: 12px !important;
    border-left: 4px solid #00FFE0 !important;
    color: #FFFFFF !important;
}

.stWarning {
    background: rgba(30, 41, 59, 0.8) !important;
    border-radius: 12px !important;
    border-left: 4px solid #ff8c00 !important;
    color: #FFFFFF !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Text Content */
.stMarkdown p {
    color: #FFFFFF !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Audio Player */
.stAudio {
    background: rgba(30, 41, 59, 0.8) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #00FFE0 !important;
}

/* Column styling */
.stColumns {
    gap: 2rem;
}

/* Subheader styling */
.stMarkdown h4 {
    color: #00FFE0 !important;
    font-weight: 600 !important;
}

/* Hide Streamlit elements */
.stDeployButton {
    display: none;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='hero-text'>✨ Welcome to ReadSmart AI</div>", unsafe_allow_html=True)

# ------------------- Session -------------------
for key in ["text_pages", "summaries", "mcqs"]:
    if key not in st.session_state:
        st.session_state[key] = []

if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False

if 'ai_activated' not in st.session_state:
    st.session_state.ai_activated = False

# ------------------- Upload -------------------
st.markdown("### 📥 Upload PDF")
pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if pdf_file:
    st.session_state.pdf_uploaded = True
    st.session_state.text_pages.clear()
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                st.session_state.text_pages.append(text.strip())
    st.success(f"✅ Extracted {len(st.session_state.text_pages)} pages.")

# ------------------- NLP Logic -------------------
def generate_questions_from_page(page_text, max_questions=3):
    prompt = f"generate questions: {page_text.strip()}"
    inputs = qg_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = qg_model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    decoded = qg_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    questions = [q.strip() for q in decoded.split("<sep>") if len(q.strip().split()) >= 4]
    return questions[:max_questions]

def extract_answer(question, context):
    try:
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except:
        return "(No answer)"

def create_mcq(question, correct_answer, context):
    words = list(set(context.split()))
    distractors = [w for w in words if w.lower() != correct_answer.lower() and len(w) > 3]
    random.shuffle(distractors)
    options = [correct_answer] + distractors[:3]
    random.shuffle(options)
    correct_option = ['A', 'B', 'C', 'D'][options.index(correct_answer)]
    return {
        "question": question,
        "options": options,
        "answer": correct_answer,
        "correct_option": correct_option
    }

def generate_mcqs(context_text):
    questions = generate_questions_from_page(context_text)
    mcqs = []
    for q in questions:
        answer = extract_answer(q, context_text)
        if not answer or len(answer.strip().split()) < 2 or answer.lower() in q.lower():
            continue
        mcq = create_mcq(q, answer, context_text)
        mcqs.append(mcq)
    return mcqs

# ------------------- Button -------------------
st.markdown("<div class='center-button'>", unsafe_allow_html=True)

if st.button("✨ ReadSmart AI", key="readsmart_button") and st.session_state.pdf_uploaded:
    st.session_state.summaries.clear()
    st.session_state.mcqs.clear()
    with st.spinner("🔍 Summarizing and generating quiz..."):
        for page in st.session_state.text_pages:
            input_length = len(page.split())
            dynamic_max = min(150, int(input_length * 0.8))
            dynamic_min = min(40, int(input_length * 0.4))
            summary = summarizer(page, max_length=dynamic_max, min_length=dynamic_min, do_sample=False)[0]['summary_text']
            st.session_state.summaries.append(summary)
            st.session_state.mcqs.extend(generate_mcqs(page))
    st.session_state.ai_activated = True
    st.success("🎉 AI Processing Complete!")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Tabs -------------------
tabs = st.tabs(["📄 Summarize", "❓ Quiz", "🔊 Text-to-Speech"])

with tabs[0]:
    st.header("📄 Summarized Pages")
    if st.session_state.ai_activated:
        for i, s in enumerate(st.session_state.summaries):
            st.subheader(f"Summary for Page {i+1}")
            st.write(s)
    else:
        st.info("Click 'ReadSmart AI' to generate summaries.")

with tabs[1]:
    st.header("❓ AI-Generated Quiz")
    if st.session_state.ai_activated:
        if st.session_state.mcqs:
            for i, mcq in enumerate(st.session_state.mcqs):
                st.markdown(f"**Q{i+1}:** {mcq['question']}")
                for idx, opt in enumerate(mcq['options']):
                    st.markdown(f"- {chr(65 + idx)}. {opt}")
                st.markdown(f"✅ **Answer:** {mcq['correct_option']}\n")
        else:
            st.warning("⚠️ No good quiz questions could be generated from the summary.")
    else:
        st.warning("Generate summaries first to unlock quiz.")

with tabs[2]:
    st.header("🔊 Listen to PDF Content Dynamically")
    if st.session_state.ai_activated:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📃 Full PDF Audio")
            full_text = "\n\n".join(st.session_state.text_pages)
            tts = gTTS(full_text)
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            st.audio(mp3_fp.getvalue(), format="audio/mp3")
        with col2:
            st.subheader("🧠 Summary Audio")
            summary_text = "\n\n".join(st.session_state.summaries)
            tts2 = gTTS(summary_text)
            mp3_fp2 = BytesIO()
            tts2.write_to_fp(mp3_fp2)
            st.audio(mp3_fp2.getvalue(), format="audio/mp3")
    else:
        st.warning("Upload and process your PDF to listen to its content.")
