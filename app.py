from flask import Flask, render_template, request, jsonify, send_from_directory
from rag_engine import ask_question, load_text_files
import os
import traceback

app = Flask(__name__)

# === HOME PAGE ===
@app.route('/')
def index():
    return render_template('index.html')


# === ENDPOINT CHAT ===
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "").strip()

    # Comando speciale: aggiornamento del database
    if user_message.lower() == "/aggiorna":
        try:
            load_text_files()
            return jsonify({
                "response": "🔄 Database aggiornato!\nI nuovi documenti e le immagini sono stati indicizzati correttamente."
            })
        except Exception as e:
            print("Errore durante /aggiorna:", e)
            traceback.print_exc()
            return jsonify({
                "response": f"⚠️ Errore durante l'aggiornamento: {str(e)}"
            })

    # Chat normale
    try:
        response = ask_question(user_message)
        return jsonify({"response": response})
    except Exception as e:
        print("Errore in /chat:", e)
        traceback.print_exc()
        return jsonify({
            "response": f"⚠️ Errore durante l'elaborazione: {str(e)}"
        })


# === SERVE LE IMMAGINI ESTRATTE ===
@app.route('/data/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join('data', 'images'), filename)


# === AVVIO SERVER (solo locale) ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
