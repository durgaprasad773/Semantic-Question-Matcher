# app.py (Uses OpenAI API with smaller embedding model)

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

# Model selection
embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    [
        "text-embedding-3-small",  # Most cost-effective: $0.02 per 1M tokens
        "text-embedding-3-large",  # Better quality: $0.13 per 1M tokens
        "text-embedding-ada-002"   # Legacy: $0.10 per 1M tokens
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

# --- CORE FUNCTIONS (with Caching) ---
@st.cache_data(show_spinner=False)
def get_embeddings(_client, texts: List[str], model: str) -> np.ndarray:
    """Get embeddings using OpenAI API with batching for efficiency"""
    if not texts or not any(text.strip() for text in texts):
        return np.array([])
    
    try:
        # OpenAI can handle batches efficiently
        response = _client.embeddings.create(
            input=texts,
            model=model
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return np.array([])

# --- UI LAYOUT ---
st.title("🔑 Semantic Question Matcher")
st.write("Enter your master questions and their subtopics. Then, enter the questions you want to match.")

# Display cost information
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
        placeholder="Format 1: Why is my animation lagging?<br>\tSUB_TOPIC_JS_PERFORMANCE\nFormat 2: How do I center a div?\tSUB_TOPIC_CSS_LAYOUT"
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
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01,
        help="Only show matches with a score equal to or higher than this value."
    )

with col_b:
    top_k = st.number_input(
        "Show Top K Matches",
        min_value=1,
        max_value=10,
        value=1,
        help="Number of best matches to show for each question"
    )

if st.button("Find Similar Questions", type="primary", use_container_width=True):
    master_questions = []
    master_subtopics = []
    delimiter1 = "<br>\tSUB_TOPIC_"
    delimiter2 = "\tSUB_TOPIC_"

    # Allow multiline or ----separated master questions
    if "---" in master_questions_text:
        master_lines = [q.strip() for q in master_questions_text.split('---') if q.strip()]
    else:
        master_lines = [line.strip() for line in master_questions_text.splitlines() if line.strip()]

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

    # Accept both line-by-line and --- separated format
    if "---" in generated_questions_text:
        generated_questions = [q.strip() for q in generated_questions_text.split('---') if q.strip()]
    else:
        generated_questions = [q.strip() for q in generated_questions_text.splitlines() if q.strip()]

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
                    # Get top K matches
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
                    
                    # Download option
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="semantic_matches.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No matches found above the specified similarity threshold.")
            else:
                st.error("Failed to generate embeddings. Please check your API key and try again.")
