import os
import openai
import requests

# === Step 1: Set your API keys ===
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["GROQ_API_KEY"] = "your_groq_api_key"

# === Step 2: Generate embeddings using OpenAI ===
def get_openai_embeddings(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",  # OpenAI supports embeddings
        input=text
    )
    return response.data[0].embedding

# === Step 3: Use Groq for chat/completion ===
def get_groq_response(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-70b",  # or mixtral-8x7b
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# === Step 4: Example Usage ===
if __name__ == "__main__":
    master_question = "==="
    user_question = "==="

    emb_master = get_openai_embeddings(master_question)
    emb_user = get_openai_embeddings(user_question)

    # Optional similarity logic
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([emb_master], [emb_user])[0][0]

    print(f"Similarity: {similarity * 100:.2f}%")

    if similarity > 0.98:
        print("✅ Perfect match")
    else:
        print("❌ Different question")

    # Generate Groq response
    print("\nGroq Response:")
    print(get_groq_response("Explain what Groq API does in simple terms"))
