import os
import streamlit as st
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List

st.cache_data.clear()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Semantic Question Matcher",
    page_icon="🔑",
    layout="wide"
)

# --- API KEY HANDLING ---
st.sidebar.title("Configuration")
st.sidebar.markdown("This tool requires your OpenAI API key to function.")
api_key_input = st.sidebar.text_input(
    "Enter your OpenAI API Key here",
    type="password",
    help="You can get your API key from platform.openai.com"
)

embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002"
    ],
    index=0,
    help="text-embedding-3-small is recommended for cost efficiency"
)

client = None
if api_key_input:
    try:
        client = OpenAI(api_key=api_key_input)
        st.sidebar.success("✓ API Key configured")
    except Exception as e:
        st.sidebar.error(f"Failed to configure API: {e}")

# --- UTILITY FUNCTIONS ---

def chunk_text(text: str, max_tokens: int = 8000) -> List[str]:
    """Split long text into smaller chunks within token limits."""
    words = text.split()
    chunks, chunk, tokens = [], [], 0
    for word in words:
        tokens += len(word) // 4  # rough token estimate
        if tokens >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk, tokens = [], 0
        chunk.append(word)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

@st.cache_data(show_spinner=False)
def get_embeddings(_client, texts: List[str], model: str) -> np.ndarray:
    """Generate embeddings safely with chunk handling."""
    if not texts or not any(text.strip() for text in texts):
        return np.array([])

    all_embeddings = []

    try:
        for text in texts:
            if len(text.split()) > 6000:  # approx token safety margin
                sub_chunks = chunk_text(text)
                sub_embeds = []
                for sub_chunk in sub_chunks:
                    response = _client.embeddings.create(input=sub_chunk, model=model)
                    sub_embeds.append(response.data[0].embedding)
                # Average chunk embeddings to represent the full text
                avg_embedding = np.mean(np.array(sub_embeds), axis=0)
                all_embeddings.append(avg_embedding)
            else:
                response = _client.embeddings.create(input=text, model=model)
                all_embeddings.append(response.data[0].embedding)
        return np.array(all_embeddings)
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return np.array([])

# --- UI LAYOUT ---
st.title("🔑 Semantic Question Matcher")
st.write("Enter your master questions and their subtopics. Then, enter the questions you want to match.")

with st.sidebar.expander("💰 Cost Information"):
    st.markdown("""
    **Embedding Model Costs:**
    - `text-embedding-3-small`: $0.02 / 1M tokens
    - `text-embedding-3-large`: $0.13 / 1M tokens
    - `text-embedding-ada-002`: $0.10 / 1M tokens
    
    **Example:** 1000 questions (~10K tokens) with 3-small costs ~$0.0002
    """)

if not client:
    st.info("Please enter a valid OpenAI API key in the sidebar to begin.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.header("Master Questions")
    master_questions_text = st.text_area(
        "Enter one entry per line using your custom format",
        height=300,
        placeholder="Format: Why is my animation lagging?<br>\tSUB_TOPIC_JS_PERFORMANCE"
    )

with col2:
    st.header("Questions to Match")
    generated_questions_text = st.text_area(
        "Enter the new questions you want to find matches for.",
        height=300,
        placeholder="e.g., My animation isn't smooth.\nWhat's the best way to center an element with CSS?"
    )

st.markdown("---")

col_a, col_b = st.columns(2)
with col_a:
    similarity_threshold = st.slider(
        "Similarity Score Threshold",
        0.0, 1.0, 0.75, 0.01,
        help="Only show matches with a score equal to or higher than this value."
    )

with col_b:
    top_k = st.number_input(
        "Show Top K Matches", 1, 10, 1,
        help="Number of best matches to show for each question."
    )

if st.button("Find Similar Questions", type="primary", use_container_width=True):
    master_questions, master_subtopics = [], []
    delimiter1, delimiter2 = "<br>\tSUB_TOPIC_", "\tSUB_TOPIC_"

    master_lines = [
        q.strip() for q in
        (master_questions_text.split('---') if '---' in master_questions_text else master_questions_text.splitlines())
        if q.strip()
    ]

    for line in master_lines:
        question, subtopic = "", "N/A"
        if delimiter1 in line:
            parts = line.split(delimiter1, 1)
            question, subtopic = parts[0].strip(), parts[1].strip()
        elif delimiter2 in line:
            parts = line.split(delimiter2, 1)
            question, subtopic = parts[0].strip(), parts[1].strip()
        else:
            question = line.strip()
        master_questions.append(question)
        master_subtopics.append(subtopic)

    generated_questions = [
        q.strip() for q in
        (generated_questions_text.split('---') if '---' in generated_questions_text else generated_questions_text.splitlines())
        if q.strip()
    ]

    if not master_questions or not generated_questions:
        st.warning("Please enter questions in both text areas.")
    else:
        with st.spinner(f"Calculating embeddings using {embedding_model}..."):
            master_embeddings = get_embeddings(client, master_questions, embedding_model)
            generated_embeddings = get_embeddings(client, generated_questions, embedding_model)

            if master_embeddings.size > 0 and generated_embeddings.size > 0:
                similarity_matrix = cosine_similarity(generated_embeddings, master_embeddings)
                results = []

                for i, gen_q in enumerate(generated_questions):
                    top_indices = np.argsort(similarity_matrix[i])[::-1][:top_k]
                    for rank, idx in enumerate(top_indices, 1):
                        score = similarity_matrix[i][idx]
                        if score >= similarity_threshold:
                            results.append({
                                "Your Question": gen_q,
                                "Rank": rank,
                                "Best Match in Master List": master_questions[idx],
                                "Subtopic": master_subtopics[idx],
                                "Similarity Score": f"{score:.3f}"
                            })

                if results:
                    st.success(f"Found {len(results)} matches above the threshold!")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results as CSV",
                        csv,
                        "semantic_matches.csv",
                        "text/csv"
                    )
                else:
                    st.info("No matches found above the specified similarity threshold.")
            else:
                st.error("Failed to generate embeddings. Please check your API key and try again.")
