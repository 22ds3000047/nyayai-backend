from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Supabase client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Create FastAPI app
app = FastAPI(title="NyayAI Backend")

# ----------- Request Schema -----------

class AskRequest(BaseModel):
    question: str
    language: str  # "English" or "Hindi"

# ----------- Helper Functions -----------

def embed_text(text: str):
    """
    Generate embedding for a given text using OpenAI
    """
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def retrieve_legal_context(question: str, limit: int = 3):
    """
    Retrieve most relevant legal chunks from Supabase using pgvector
    """
    question_embedding = embed_text(question)

    result = supabase.rpc(
        "match_legal_chunks",
        {
            "query_embedding": question_embedding,
            "match_count": limit
        }
    ).execute()

    return result.data if result.data else []


def generate_answer(question: str, context: list, language: str):
    """
    Generate a safe, structured legal information answer
    """
    if not context:
        return (
            "Answer:\n"
            "I’m not sure based on the available information.\n\n"
            "Sources:\n"
            "No relevant legal sources found.\n\n"
            "Disclaimer:\n"
            "This is legal information, not legal advice. Please consult a licensed advocate."
        )

    context_text = "\n\n".join(
        f"{row['act_name']} {row['section']}: {row['content']}"
        for row in context
    )

    prompt = f"""
You are NyayAI, a legal information assistant for India.

Rules:
- Use ONLY the provided legal context
- Cite the relevant Act and Section
- Do NOT give legal advice
- Do NOT predict outcomes
- If unsure, clearly say you do not know
- Answer in {language}

Legal Context:
{context_text}

User Question:
{question}

Respond EXACTLY in this format:

Answer:
<plain language explanation>

Sources:
• Act name – Section number

Disclaimer:
This is legal information, not legal advice. Please consult a licensed advocate.
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ----------- API Endpoint -----------

@app.post("/ask")
def ask_legal_question(req: AskRequest):
    context = retrieve_legal_context(req.question)
    answer = generate_answer(req.question, context, req.language)

    return {
        "response": answer
    }
