import os
import io
import shutil
import gc
from openai import OpenAI
import fitz  # PyMuPDF
from PIL import Image
import docx
import lancedb
import numpy as np
import pyarrow as pa

# === CONFIGURAZIONE BASE ===
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "lancedb")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# === GESTIONE LANCEDB ===
def connect_lancedb():
    """Crea o riapre il database LanceDB, ricreandolo se danneggiato."""
    schema = pa.schema([
        ("filename", pa.string()),
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
        # Se il DB è corrotto o mancante, lo ricrea da zero
        shutil.rmtree(DB_PATH, ignore_errors=True)
        os.makedirs(DB_PATH, exist_ok=True)
        db = lancedb.connect(DB_PATH)
        db.create_table("documents", schema=schema, mode="overwrite")
        table = db.open_table("documents")
        return db, table


db, table = connect_lancedb()


# === ESTRAZIONE CONTENUTI ===
def extract_text_from_pdf(pdf_path):
    """Estrae testo da PDF e salva le immagini correlate."""
    text = ""
    images = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text += page.get_text("text") + "\n"
            for img_index, img in enumerate(page.get_images(full=True)):
                if len(images) >= 5:  # ✅ Limite massimo di 5 immagini per documento
                    break
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image = Image.open(io.BytesIO(image_bytes))
                image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_{page_index+1}_{img_index+1}.{image_ext}"
                save_path = os.path.join(IMAGE_DIR, image_filename)
                image.save(save_path)
                images.append(save_path)
    return text.strip(), images


def extract_text_from_docx(docx_path):
    """Estrae testo dai file Word (.docx)."""
    document = docx.Document(docx_path)
    return "\n".join([p.text for p in document.paragraphs]).strip()


# === INDICIZZAZIONE DOCUMENTI ===
def load_text_files():
    """Legge tutti i file nella cartella /data e li indicizza in LanceDB."""
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

        text = ""
        if filename.lower().endswith(".pdf"):
            text, _ = extract_text_from_pdf(filepath)
        elif filename.lower().endswith(".docx"):
            text = extract_text_from_docx(filepath)
        elif filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

        if not text.strip():
            continue

        # ✅ Limita la lunghezza del testo per ridurre memoria ed embedding
        text_snippet = text[:2000]

        # Crea embedding per il testo
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text_snippet
        ).data[0].embedding

        table.add([{"filename": filename, "content": text_snippet, "vector": emb}])
        print(f"Indicizzato: {filename}")

        # ✅ Libera la memoria dopo ogni file
        gc.collect()


# === FUNZIONE PRINCIPALE ===
def ask_question(query):
    """Cerca nei documenti e genera una risposta (testo + analisi visiva)."""
    global db, table

    # Aggiorna database se serve
    load_text_files()

    # Crea embedding della domanda
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Ricerca semantica nei documenti
    try:
        results = table.search(query_emb).limit(3).to_list()
    except Exception:
        db, table = connect_lancedb()
        results = []

    # Costruisce il contesto dai documenti trovati
    if not results:
        context = "Nessun documento rilevante trovato."
    else:
        context = "\n\n".join([r["content"][:1500] for r in results])

    # Aggiunge immagini correlate (massimo 2 per evitare saturazione RAM)
    images = []
    for file in os.listdir(IMAGE_DIR):
        if any(r["filename"].split('.')[0] in file for r in results):
            images.append(os.path.join(IMAGE_DIR, file))
    images = images[:2]  # ✅ Limita a 2 immagini

    # Prepara i messaggi per GPT-4o
    messages = [
        {
            "role": "system",
            "content": (
                "Sei un assistente che risponde in base ai documenti e alle immagini "
                "presenti nella base dati. Fornisci risposte chiare e concise."
            )
        },
        {"role": "user", "content": f"Contesto:\n{context}\n\nDomanda: {query}"}
    ]

    if images:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Analizza anche queste immagini correlate:"},
                *[
                    {"type": "image_url", "image_url": f"file://{os.path.abspath(img)}"}
                    for img in images
                ]
            ]
        })

    # Genera risposta multimodale
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    # ✅ Libera memoria prima di restituire la risposta
    gc.collect()

    return completion.choices[0].message.content.strip()
