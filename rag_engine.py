import os
from openai import OpenAI
import fitz  # PyMuPDF

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path):
    """Estrae tutto il testo da un PDF."""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def ask_question(query):
    data_path = "data"
    context = ""

    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)

        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                context += f.read() + "\n"

        elif filename.endswith(".pdf"):
            context += extract_text_from_pdf(filepath) + "\n"

    if not context.strip():
        context = "Nessun documento disponibile nella cartella data."

    prompt = f"Usa il seguente contesto per rispondere:\n{context}\n\nDomanda: {query}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sei un assistente utile e preciso che risponde solo in base ai documenti forniti."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()
