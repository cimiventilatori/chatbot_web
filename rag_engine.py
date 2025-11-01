import os
import fitz  # PyMuPDF
import docx
from openai import OpenAI
import lancedb

# Inizializza il client OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Percorsi
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "lancedb")

# Crea o apre il database LanceDB
os.makedirs(DB_PATH, exist_ok=True)
db = lancedb.connect(DB_PATH)
table_name = "documents"

# Crea (se non esiste) una tabella per i documenti
if table_name not in db.table_names():
    db.create_table(table_name, data=[], mode="overwrite")
table = db.open_table(table_name)


# === FUNZIONI DI ESTRAZIONE TESTO ===

def extract_text_from_pdf(path):
    """Estrae testo da un file PDF."""
    text = ""
    with fitz.open(path) as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text.strip()


def extract_text_from_docx(path):
    """Estrae testo da un file Word (.docx)."""
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs]).strip()


def load_text_files():
    """Legge e indicizza tutti i file PDF, DOCX, TXT nella cartella data."""
    files = os.listdir(DATA_DIR)
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)

        # Skippa la cartella del DB LanceDB
        if os.path.isdir(filepath):
            continue

        # Controlla se già indicizzato
        if table.search("filename", "==", filename).count() > 0:
            continue

        # Estrae testo a seconda del tipo di file
        text = ""
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        elif filename.lower().endswith(".docx"):
            text = extract_text_from_docx(filepath)
        elif filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()

        if not text:
            continue

        # Crea embedding per il testo
        embedding = client.embeddings.create(
            input=text[:3000],  # limitiamo per non superare token max
            model="text-embedding-3-small"
        ).data[0].embedding

        # Salva nel database
        table.add([
            {"filename": filename, "content": text, "vector": embedding}
        ])
        print(f"Indicizzato: {filename}")


# === FUNZIONE PRINCIPALE ===

def ask_question(query):
    """Esegue una ricerca semantica e genera una risposta."""
    # Aggiorna database se ci sono nuovi documenti
    load_text_files()

    # Crea embedding della domanda
    query_embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Ricerca nel database i documenti più rilevanti
    results = table.search(query_embedding).limit(3).to_list()

    if not results:
        context = "Nessun documento trovato nella base dati."
    else:
        context = "\n\n".join([r["content"][:1500] for r in results])

    # Genera risposta basata sul contesto trovato
    prompt = f"Contesto dai documenti:\n{context}\n\nDomanda: {query}\nRisposta:"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sei un assistente che risponde in base ai documenti indicizzati."},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content.strip()
