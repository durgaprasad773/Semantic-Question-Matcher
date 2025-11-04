import os
import openai
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# 🔐 Add your API keys manually
# ============================
OPENAI_API_KEY = "sk-xxxxx_your_openai_key_here"
GROQ_API_KEY = "gsk_xxxxx_your_groq_key_here"

# ============================
# ⚙️ Configure APIs
# ============================
openai.api_key = OPENAI_API_KEY


# ============================
# 🧩 1. Generate Embeddings (OpenAI)
# ============================
def get_openai_embeddings(text):
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",  # Best for low-cost similarity
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print("Error getting embeddings:", e)
        return None


# ============================
# 🚀 2. Generate Response (Groq)
# ============================
def get_groq_response(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-70b",  # You can use mixtral-8x7b if preferred
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("Error getting Groq response:", e)
        return None


# ============================
# 🔍 3. Compare Two Questions
# ============================
def compare_questions(master_q, user_q):
    # Exact string match handling
    if master_q.strip() == user_q.strip():
        return 1.0  # Perfect 100% match

    emb_master = get_openai_embeddings(master_q)
    emb_user = get_openai_embeddings(user_q)

    if emb_master is None or emb_user is None:
        print("❌ Failed to get embeddings. Please check API keys or input size.")
        return 0.0

    similarity = cosine_similarity([emb_master], [emb_user])[0][0]
    return similarity


# ============================
# 🧠 4. Example Run
# ============================
if __name__ == "__main__":
    master_question = "==="
    user_question = "==="

    similarity = compare_questions(master_question, user_question)
    print(f"🔹 Similarity Score: {similarity * 100:.2f}%")

    if similarity >= 0.98:
        print("✅ 100% Perfect Match (Same Question)")
    elif similarity >= 0.85:
        print("🟢 Highly Similar")
    else:
        print("❌ Different Question")

    # Optional: Get Groq explanation
    prompt = f"Explain how similar these two questions are:\nQ1: {master_question}\nQ2: {user_question}"
    response = get_groq_response(prompt)
    if response:
        print("\n🤖 Groq Response:")
        print(response)
