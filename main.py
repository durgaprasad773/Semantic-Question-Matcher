import os
import streamlit as st
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List
import re
import time

st.cache_data.clear()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Semantic Question Matcher",
    page_icon="🔑",
    layout="wide"
)

# --- API KEY HANDLING ---
st.sidebar.title("Configuration")
st.sidebar.markdown("This tool requires your Groq API key to function.")
api_key_input = st.sidebar.text_input(
    "Enter your Groq API Key here",
    type="password",
    help="You can get your API key from console.groq.com"
)

# Placeholder model names for embeddings (update when Groq supports them)
embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    [
        "groq-embedding-small",   # placeholder
        "groq-embedding-large",   # placeholder
        "groq-embedding-legacy"   # placeholder
    ],
    index=0,
    help="Select embedding model (Groq) – update when actual models available"
)

client_headers = {}
if api_key_input:
    client_headers = {
        "Authorization": f"Bearer {api_key_input}",
        "Content-Type": "application/json"
    }
    st.sidebar.success("✓ API Key configured")

def chunk_text(text: str, max_tokens: int = 8000) -> List[str]:
    words = text.split()
    chunks, chunk, tokens = [], [], 0
    for word in words:
        tokens += len(word) / 4
        if tokens >= max_tokens - 500:
            chunks.append(" ".join(chunk))
            chunk, tokens = [], 0
        chunk.append(word)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

@st.cache_data(show_spinner=False)
def get_embeddings(texts: List[str], model: str) -> np.ndarray:
    if not texts or not any(text.strip() for text in texts):
        return np.array([])

    all_embeddings = []
    # Placeholder token limit; update when Groq publishes embedding limit
    limit = 8192

    try:
        for text in texts:
            text = text.strip()
            if not text:
                continue

            est_tokens = len(text) / 4
            if est_tokens > limit:
                st.warning(f"⚠️ A long question was automatically split (approx {int(est_tokens)} tokens).")
                chunks = chunk_text(text, limit)
                chunk_embeddings = []
                for c in chunks:
                    # Example endpoint; update according to Groq embedding API when available
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/embeddings",
                        headers=client_headers,
                        json={
                            "model": model,
                            "input": c
                        }
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # adapt depending on API response structure
                    embedding_vec = data["data"][0]["embedding"]
                    chunk_embeddings.append(embedding_vec)
                    time.sleep(0.2)
                avg_embedding = np.mean(np.array(chunk_embeddings), axis=0)
                all_embeddings.append(avg_embedding)
            else:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/embeddings",
                    headers=client_headers,
                    json={
                        "model": model,
                        "input": text
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                embedding_vec = data["data"][0]["embedding"]
                all_embeddings.append(embedding_vec)
                time.sleep(0.2)

        return np.array(all_embeddings)
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return np.array([])

# --- UI LAYOUT ---
st.title("🔑 Semantic Question Matcher (Groq version)")
st.write("Enter your master questions + subtopics, then the questions you want to match.")

with st.sidebar.expander("💰 Cost & Notes"):
    st.markdown("""  
    ⚠️ Note: Groq embedding model support is **currently not confirmed**.  
    Please verify model availability and endpoint.  
    """)

if not api_key_input:
    st.info("Please enter a valid Groq API key in the sidebar to begin.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.header("Master Questions")
    master_questions_text = st.text_area(
        "Enter one entry per line. Format: Question <tab> SUB_TOPIC_xxx",
        height=300,
        placeholder="Why is my animation lagging?\tSUB_TOPIC_JS_PERFORMANCE"
    )

with col2:
    st.header("Questions to Match")
    generated_questions_text = st.text_area(
        "Enter new questions you want to match.",
        height=300,
        placeholder="My animation isn't smooth.\nHow to center a div with CSS?"
    )

st.markdown("---")

col_a, col_b = st.columns(2)
with col_a:
    similarity_threshold = st.slider(
        "Similarity Score Threshold", 0.0, 1.0, 0.75, 0.01,
        help="Only show matches with score >= this threshold."
    )
with col_b:
    top_k = st.number_input(
        "Show Top K Matches", 1, 10, 1,
        help="Number of best matches for each question."
    )

if st.button("Find Similar Questions", type="primary", use_container_width=True):
    master_questions, master_subtopics = [], []
    delimiter = "\tSUB_TOPIC_"

    master_lines = [q.strip() for q in master_questions_text.splitlines() if q.strip()]
    for line in master_lines:
        if delimiter in line:
            parts = line.split(delimiter, 1)
            question = parts[0].strip()
            subtopic = parts[1].strip()
        else:
            question = line.strip()
            subtopic = "N/A"
        master_questions.append(question)
        master_subtopics.append(subtopic)

    generated_questions = [q.strip() for q in generated_questions_text.splitlines() if q.strip()]

    if not master_questions or not generated_questions:
        st.warning("Please enter both master questions and questions to match.")
    else:
        with st.spinner(f"Calculating embeddings using {embedding_model}..."):
            master_embeddings = get_embeddings(master_questions, embedding_model)
            generated_embeddings = get_embeddings(generated_questions, embedding_model)

            if master_embeddings.size > 0 and generated_embeddings.size > 0:
                similarity_matrix = cosine_similarity(generated_embeddings, master_embeddings)
                results = []
                normalize = lambda s: re.sub(r"\s+", " ", s.strip().lower())

                for i, gen_q in enumerate(generated_questions):
                    top_indices = np.argsort(similarity_matrix[i])[::-1][:top_k]
                    for rank, idx in enumerate(top_indices, 1):
                        score = similarity_matrix[i][idx]
                        if normalize(gen_q) == normalize(master_questions[idx]):
                            score = 1.0
                        if score >= similarity_threshold:
                            results.append({
                                "Your Question": gen_q,
                                "Rank": rank,
                                "Best Match in Master List": master_questions[idx],
                                "Subtopic": master_subtopics[idx],
                                "Similarity Score": f"{score:.3f}"
                            })

                if results:
                    st.success(f"Found {len(results)} matches above threshold!")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download Results as CSV", csv, "semantic_matches.csv", "text/csv")
                else:
                    st.info("No matches found above the specified similarity threshold.")
            else:
                st.error("Failed to generate embeddings. Check your API key and model availability.")
