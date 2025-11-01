import os
import io
from openai import OpenAI
import fitz  # PyMuPDF
from PIL import Image

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

def extract_text_and_images_from_pdf(pdf_path):
    """Estrae testo e immagini da un PDF, salvando le immagini come file PNG."""
    text = ""
    images = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text += page.get_text("text") + "\n"
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image = Image.open(io.BytesIO(image_bytes))
                image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_{page_index+1}_{img_index+1}.{image_ext}"
                image.save(os.path.join(IMAGE_DIR, image_filename))
                images.append(os.path.join(IMAGE_DIR, image_filename))
    return text, images

def extract_text_from_docx(doc_path):
    """Estrae testo da un file Word (.docx)."""
    import docx
    doc = docx.Document(doc_path)
    return "\n".join([p.text for p in doc.paragraphs])

def load_all_documents():
    """Raccoglie testo e immagini da tutti i file nella cartella data."""
    context = ""
    all_images = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.isdir(filepath):
            continue
        if filename.lower().endswith(".pdf"):
            text, imgs = extract_text_and_images_from_pdf(filepath)
            context += f"\n\n---\nFile: {filename}\n{text}"
            all_images.extend(imgs)
        elif filename.lower().endswith(".docx"):
            text = extract_text_from_docx(filepath)
            context += f"\n\n---\nFile: {filename}\n{text}"
        elif filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                context += f"\n\n---\nFile: {filename}\n{f.read()}"
    return context.strip(), all_images

def ask_question(query):
    """Invia a GPT-4o testo e immagini dei documenti per ottenere una risposta multimodale."""
    context, images = load_all_documents()

    messages = [
        {"role": "system", "content": "Sei un assistente che analizza documenti testuali e immagini, fornendo risposte precise e concise."},
        {"role": "user", "content": f"Contesto:\n{context}\n\nDomanda: {query}"}
    ]

    # Aggiunge le immagini come input visivo
    if images:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Analizza anche queste immagini correlate ai documenti:"},
                *[{"type": "image_url", "image_url": f"file://{os.path.abspath(img)}"} for img in images[:5]]  # massimo 5 immagini
            ]
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    return response.choices[0].message.content.strip()
