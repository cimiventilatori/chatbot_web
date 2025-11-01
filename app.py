from flask import Flask, render_template, request, jsonify
from rag_engine import ask_question, load_text_files

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "").strip()

    # Comando speciale per aggiornare l'indicizzazione
    if user_message.lower() == "/aggiorna":
        try:
            load_text_files()
            return jsonify({"response": "🔄 Database aggiornato! I nuovi documenti sono stati indicizzati correttamente."})
        except Exception as e:
            return jsonify({"response": f"⚠️ Errore durante l'aggiornamento: {str(e)}"})

    # Risposta normale del chatbot
    try:
        response = ask_question(user_message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"⚠️ Errore durante l'elaborazione: {str(e)}"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
