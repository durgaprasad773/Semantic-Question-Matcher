import streamlit as st
import openai
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# ⚙️ Streamlit Page Setup
# ============================
st.set_page_config(page_title="Semantic Similarity Checker", page_icon="🔍", layout="centered")
st.title("🔍 Semantic Question Matcher (Groq + OpenAI)")

st.markdown("""
This app compares **two questions** using OpenAI embeddings and provides an explanation using **Groq AI**.  
Enter both API keys below and your questions to get the similarity result.
""")

# ============================
# 🔐 API Keys Input (User provides)
# ============================
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password", help="Required for generating embeddings.")
groq_api_key = st.text_input("⚡ Groq API Key", type="password", help="Required for getting explanations.")

# Initialize clients only if keys are provided
if openai_api_key:
    openai.api_key = openai_api_key


# ============================
# 🧩 Helper Functions
# ============================

def get_openai_embeddings(text):
    """Generate OpenAI embeddings for a given text."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"❌ Error getting embeddings: {e}")
        return None


def get_groq_response(prompt):
    """Get a Groq response (chat completion)."""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-70b",
            "messages": [
                {"role": "system", "content": "You are an expert semantic evaluator."},
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"❌ Error getting Groq response: {e}")
        return None


def compare_questions(master_q, user_q):
    """Compare two questions and return a similarity score."""
    if master_q.strip() == user_q.strip():
        return 1.0  # Perfect match

    emb_master = get_openai_embeddings(master_q)
    emb_user = get_openai_embeddings(user_q)

    if emb_master is None or emb_user is None:
        return None

    similarity = cosine_similarity([emb_master], [emb_user])[0][0]
    return similarity


# ============================
# 🧠 User Inputs
# ============================
st.subheader("📝 Enter Your Questions")

col1, col2 = st.columns(2)

with col1:
    master_question = st.text_area("Master Question", height=120, placeholder="e.g., Why is my animation lagging?")
with col2:
    user_question = st.text_area("Your Question", height=120, placeholder="e.g., My animation isn't smooth.")

# ============================
# 🚀 Compare Button
# ============================
if st.button("🔎 Compare Questions", type="primary", use_container_width=True):
    if not openai_api_key or not groq_api_key:
        st.warning("⚠️ Please enter both OpenAI and Groq API keys.")
    elif not master_question or not user_question:
        st.warning("⚠️ Please enter both questions.")
    else:
        with st.spinner("Calculating similarity..."):
            similarity = compare_questions(master_question, user_question)

        if similarity is not None:
            similarity_percent = similarity * 100
            st.success(f"🔹 Similarity Score: **{similarity_percent:.2f}%**")

            if similarity >= 0.98:
                st.markdown("✅ **Perfect Match (100%)** — Questions are identical.")
            elif similarity >= 0.85:
                st.markdown("🟢 **Highly Similar** — Questions mean almost the same.")
            elif similarity >= 0.6:
                st.markdown("🟡 **Moderately Similar** — Some overlap in meaning.")
            else:
                st.markdown("🔴 **Low Similarity** — Different questions.")

            # Generate Groq explanation
            with st.spinner("Generating Groq explanation..."):
                prompt = f"Compare these two questions semantically and explain their similarity:\n\nQ1: {master_question}\nQ2: {user_question}"
                explanation = get_groq_response(prompt)

            if explanation:
                st.subheader("🤖 Groq Explanation")
                st.write(explanation)
        else:
            st.error("Failed to compute similarity. Please check API keys or input size.")

# ============================
# ℹ️ Footer
# ============================
st.markdown("---")
st.caption("Built with ❤️ using OpenAI for embeddings and Groq for responses.")
