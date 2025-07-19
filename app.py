import streamlit as st
import pdfplumber
from transformers import pipeline
from gtts import gTTS
from io import BytesIO

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.set_page_config(page_title="📘 ReadSmart", layout="wide")

st.title("📘 ReadSmart – AI-powered PDF Understanding Tool")

# Tabs
tabs = st.tabs(["📥 Upload PDF", "📄 Summarize", "❓ Quiz (Simulated)", "🔊 Text-to-Speech"])

# Global session state
if 'text_pages' not in st.session_state:
    st.session_state.text_pages = []

if 'summaries' not in st.session_state:
    st.session_state.summaries = []

# 1. PDF Upload
with tabs[0]:
    st.header("📥 Upload Your PDF")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if pdf_file:
        st.success("PDF uploaded successfully!")
        st.session_state.text_pages.clear()
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    st.session_state.text_pages.append(text.strip())
        st.write(f"✅ Extracted {len(st.session_state.text_pages)} pages.")

# 2. Summarize
with tabs[1]:
    st.header("📄 Summarize Extracted Content")
    if st.session_state.text_pages:
        if st.button("🧠 Generate Summaries"):
            st.session_state.summaries.clear()
            with st.spinner("Summarizing..."):
                for page in st.session_state.text_pages[:2]:  # Just first 2 for now
                    summary = summarizer(page, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
                    st.session_state.summaries.append(summary)
            st.success("Summaries generated!")
        
        for i, s in enumerate(st.session_state.summaries):
            st.subheader(f"📄 Summary for Page {i+1}")
            st.write(s)
    else:
        st.info("Please upload and extract PDF content first.")

# 3. Simulated Quiz
with tabs[2]:
    st.header("❓ Auto-Generated Questions (Simulated)")
    if st.session_state.summaries:
        for i, summary in enumerate(st.session_state.summaries[:1]):
            st.subheader(f"Page {i+1} Sample Quiz")
            st.markdown(f"""
            **Q:** What is the main idea of the summary?

            - A. Random fact  
            - B. {summary.split('.')[0].strip()} ✅  
            - C. Unrelated sentence  
            - D. General statement

            ✅ Correct Answer: B
            """)
    else:
        st.warning("Summarize content to view quiz.")

# 4. TTS
with tabs[3]:
    st.header("🔊 Listen to the Summary")
    if st.session_state.summaries:
        summary_text = " ".join(st.session_state.summaries[:1])  # One page summary for now
        tts = gTTS(summary_text)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        st.audio(mp3_fp.getvalue(), format="audio/mp3")
    else:
        st.warning("Please generate summaries first.")
