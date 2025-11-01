import os
from openai import OpenAI

# Inizializza il client OpenAI con la chiave API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_question(query):
    data_path = "data"
    context = ""

    # Legge tutti i file .txt nella cartella data
    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                context += f.read() + "\n"

    prompt = f"Usa il seguente contesto per rispondere:\n{context}\n\nDomanda: {query}"

    # Nuova sintassi per la chiamata API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sei un assistente utile e preciso."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()
