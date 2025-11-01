import os
import io
import shutil
import gc
from openai import OpenAI
import fitz  # PyMuPDF
import docx
import lancedb
import numpy as np
import pyarrow as pa
import textwrap

# === CONFIGURAZIONE BASE ===
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "lancedb")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# === GESTIONE LANCEDB ===
def connect_lancedb():
    """Crea o riapre il database LanceDB con schema predefinito."""
    schema = pa.schema([
        ("filename", pa.string()),
        ("chunk_id", pa.int32()),
        ("content", pa.string()),
        ("vector", pa.list_(pa.float32()))
    ])

    try:
        db = lancedb.connect(DB_PATH)
        table_name = "documents"

        if table_name not in db.table_names():
            db.create_table(table_name, schema=schema, mode="overwrite")

        table = db.open_table(table_name)
        return db, table

    except Exception:
        shutil.rmtree(DB_PATH, ignore_errors=True)
        os.makedirs(DB_PATH, exist_ok=True)
        db = lancedb.connect(DB_PATH)
        db.create_table("documents", schema=schema, mode="overwrite")
        table = db.open_table("documents")
        return db, table


db, table = connect_lancedb()


# === ESTRAZIONE TESTO DAI DOCUMENTI ===
def extract_text_from_pdf(pdf_path):
    """Estrae il testo da un PDF, pagina per pagina."""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text("text") + "\n"
    return text.strip()


def extract_text_from_docx(docx_path):
    """Estrae testo dai file Word (.docx)."""
    document = docx.Document(docx_path)
    return "\n".join([p.text for p in document.paragraphs]).strip()


# === FUNZIONE DI CHUNKING ===
def chunk_text(text, max_length=1200):
    """
    Divide il testo in piccoli blocchi per l'embedding.
    max_length = numero massimo di caratteri per chunk.
    """
    chunks = textwrap.wrap(text, width=max_length, break_long_words=False)
    return chunks


# === INDICIZZAZIONE DOCUMENTI ===
def load_text_files():
    """Indicizza tutti i documenti di testo, PDF e Word in chunk."""
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.isdir(filepath):
            continue

        # Evita duplicati
        try:
            if table.search("filename", "==", filename).count() > 0:
                continue
        except Exception:
            pass

        # Estrazione testo
        text = ""
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        elif filename.lower().endswith(".docx"):
            text = extract_text_from_docx(filepath)
        elif filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

        if not text.strip():
            continue

        # Suddivide il testo in chunk
        chunks = chunk_text(text)

        # Indicizza ogni chunk
        for i, chunk in enumerate(chunks):
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            ).data[0].embedding

            table.add([{
                "filename": filename,
                "chunk_id": i,
                "content": chunk,
                "vector": emb
            }])

        print(f"Indicizzato: {filename} ({len(chunks)} chunk)")
        gc.collect()


# === RICERCA E RISPOSTA ===
def ask_question(query):
    """Cerca nei documenti e genera una risposta basata sui chunk rilevanti."""
    global db, table

    # Aggiorna database se serve
    load_text_files()

    # Embedding della domanda
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Cerca chunk rilevanti
    try:
        results = table.search(query_emb).limit(5).to_list()
    except Exception:
        db, table = connect_lancedb()
        results = []

    if not results:
        context = "Nessun contenuto rilevante trovato."
    else:
        context = "\n\n".join([r["content"] for r in results])

    # Prepara i messaggi per GPT-4o
    messages = [
        {
            "role": "system",
            "content": (
                "Sei un assistente che risponde alle domande "
                "utilizzando solo le informazioni fornite dai documenti caricati."
            )
        },
        {"role": "user", "content": f"Contesto:\n{context}\n\nDomanda: {query}"}
    ]

    # Chiamata al modello
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    gc.collect()
    return completion.choices[0].message.content.strip()
