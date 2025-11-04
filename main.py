import os
import time
import streamlit as st
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# --- Streamlit Setup ---
st.set_page_config(page_title="Semantic Similarity App", page_icon="🧠", layout="wide")
st.title("🧠 Semantic Similarity & Groq Explanation App")
st.write("Compare two questions semantically using OpenAI embeddings and get explanations from Groq.")

# --- Sidebar for API Keys ---
st.sidebar.header("🔐 API Configuration")
openai_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
groq_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

# Initialize clients only when keys are entered
if openai_key:
    client = OpenAI(api_key=openai_key)
else:
    client = None

if not openai_key or not groq_key:
    st.warning("⚠️ Please provide both OpenAI and Groq API keys to proceed.")
    st.stop()

# --- Groq Response Function ---
def get_groq_response(prompt):
    """Get explanation from Groq API."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-70b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that explains question similarity clearly."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        res = requests.post(url, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"❌ Error fetching Groq response: {e}")
        return None


# --- Auto-Chunked Embedding Function ---
def get_openai_embeddings(text, model="text-embedding-3-small", max_chunk_tokens=8000):
    """Automatically splits long text and computes averaged embeddings."""
    import math

    try:
        if not text.strip():
            return None

        # Rough token estimation (1 token ≈ 4 chars)
        token_estimate = len(text) / 4
        if token_estimate <= max_chunk_tokens:
            resp = client.embeddings.create(input=text, model=model)
            return np.array(resp.data[0].embedding)

        # Split text into safe chunks
        words = text.split()
        chunks = []
        chunk = []
        tokens = 0
        for word in words:
            tokens += len(word) / 4
            if tokens >= max_chunk_tokens:
                chunks.append(" ".join(chunk))
                chunk, tokens = [], 0
            chunk.append(word)
        if chunk:
            chunks.append(" ".join(chunk))

        st.info(f"⚙️ Text automatically split into {len(chunks)} chunks to fit token limit.")

        embeddings = []
        for ch in chunks:
            resp = client.embeddings.create(input=ch, model=model)
            embeddings.append(resp.data[0].embedding)
            time.sleep(0.2)  # Avoid rate limiting

        avg_embedding = np.mean(np.array(embeddings), axis=0)
        return avg_embedding

    except Exception as e:
        st.error(f"❌ Error getting embeddings: {e}")
        return None


# --- Input Fields ---
col1, col2 = st.columns(2)
with col1:
    q1 = st.text_area("🧩 Enter Master Question", height=200, placeholder="e.g., Why is my animation lagging?")
with col2:
    q2 = st.text_area("🎯 Enter Question to Compare", height=200, placeholder="e.g., My animation feels slow.")

if st.button("🔍 Compare Questions", type="primary", use_container_width=True):
    if not q1.strip() or not q2.strip():
        st.warning("Please fill both questions before comparing.")
        st.stop()

    with st.spinner("Calculating embeddings and similarity..."):
        # ✅ Handle exact matches instantly
        if q1.strip().lower() == q2.strip().lower():
            similarity = 1.0
        else:
            emb1 = get_openai_embeddings(q1)
            emb2 = get_openai_embeddings(q2)

            if emb1 is None or emb2 is None:
                st.error("Failed to compute embeddings. Check your API keys or inputs.")
                st.stop()

            similarity = cosine_similarity([emb1], [emb2])[0][0]

        # Display similarity
        st.subheader("🔹 Similarity Result")
        st.write(f"**Similarity Score:** {similarity * 100:.2f}%")

        if similarity >= 0.98:
            st.success("✅ Perfect Match (Identical or nearly identical meaning).")
        elif similarity >= 0.85:
            st.info("🟢 Highly Similar Questions.")
        elif similarity >= 0.65:
            st.warning("🟠 Somewhat Similar but Different Contexts.")
        else:
            st.error("❌ Different Questions.")

        # Groq Explanation
        with st.spinner("Getting explanation from Groq..."):
            prompt = f"Explain briefly how similar these two questions are:\nQ1: {q1}\nQ2: {q2}\nSimilarity: {similarity*100:.2f}%"
            explanation = get_groq_response(prompt)
            if explanation:
                st.markdown("### 🤖 Groq Explanation")
                st.write(explanation)
