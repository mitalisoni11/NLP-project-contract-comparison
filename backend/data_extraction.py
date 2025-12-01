import os
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from openai import OpenAI
import tiktoken


PDF_FOLDER = "backend/uploads/wipo_documents"
INDEX_NAME = "wipo-index"

MODEL = "text-embedding-3-small"
TOKEN_LIMIT = 400  # keep chunks small => safe Pinecone metadata

OPENAI_API_KEY = "YOUR_OPENAI_KEY"
PINECONE_API_KEY = "YOUR_PINECONE_KEY"


def load_pdf(path: str) -> str:
    """Extract the full text from a PDF."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def token_chunk(text, max_tokens=TOKEN_LIMIT):
    """Split text into token-limited chunks using tiktoken."""
    enc = tiktoken.encoding_for_model(MODEL)
    tokens = enc.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        token_slice = tokens[start:end]
        chunk_text = enc.decode(token_slice)

        # avoid empty chunks
        if chunk_text.strip():
            chunks.append(chunk_text)

        start = end

    return chunks


def embed_chunks(chunks):
    """Generate embeddings for text chunks."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        model=MODEL,
        input=chunks
    )

    return [d.embedding for d in response.data]


def upload_to_pinecone(index, file_name, country, chunks, embeddings):
    """Upload embeddings with safe metadata including text."""
    vectors = []

    for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):

        vectors.append({
            "id": f"{file_name}-{i}",
            "values": emb,
            "metadata": {
                "file_name": file_name,
                "chunk_id": i,
                "country": country,
                "text": chunk_text  # SAFE because chunks are tiny (<1kb)
            }
        })

    index.upsert(vectors=vectors)
    print(f"Uploaded {len(vectors)} chunks for {file_name}")


def main():

    # 1. Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 2. Create index if needed
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # embedding-3-small => 1536 dims
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")

    index = pc.Index(INDEX_NAME)

    # 3. Process all PDFs
    for file in os.listdir(PDF_FOLDER):
        if not file.endswith(".pdf"):
            continue

        print(f"\nProcessing: {file}")

        full_path = os.path.join(PDF_FOLDER, file)

        # Extract PDF text
        text = load_pdf(full_path)

        # Token-based chunking
        chunks = token_chunk(text)

        # Embed
        embeddings = embed_chunks(chunks)

        # Upload
        upload_to_pinecone(
            index=index,
            file_name=file,
            country="Italy",
            chunks=chunks,
            embeddings=embeddings
        )

    print("\n✓ DONE — All PDFs processed successfully.")


if __name__ == "__main__":
    main()
