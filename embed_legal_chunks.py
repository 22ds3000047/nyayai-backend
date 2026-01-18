import os
from supabase import create_client
import openai
from dotenv import load_dotenv

load_dotenv()

# Setup keys
openai.api_key = os.getenv("OPENAI_API_KEY")

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def embed_text(text: str):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def main():
    # Fetch rows without embeddings
    rows = supabase.table("legal_chunks") \
        .select("id, content") \
        .is_("embedding", None) \
        .execute()

    if not rows.data:
        print("✅ No rows to embed. All done.")
        return

    print(f"Found {len(rows.data)} rows to embed...")

    for row in rows.data:
        embedding = embed_text(row["content"])

        supabase.table("legal_chunks") \
            .update({"embedding": embedding}) \
            .eq("id", row["id"]) \
            .execute()

        print(f"Embedded row {row['id']}")

    print("🎉 All embeddings generated successfully.")

if __name__ == "__main__":
    main()
