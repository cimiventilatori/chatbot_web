from flask import Flask, render_template, request, jsonify
from rag_engine import ask_question, load_text_files, connect_lancedb
import shutil
import os
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "").strip().lower()

    # === Comando per aggiornare i documenti ===
    if user_message == "/aggiorna":
        try:
            load_text_files()
            return jsonify({
                "response": "🔄 Database aggiornato!\nI nuovi documenti sono stati indicizzati correttamente."
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({"response": f"⚠️ Errore durante l'aggiornamento: {str(e)}"})

    # === Comando per resettare completamente il database ===
    if user_message == "/reset":
        try:
            db_path = os.path.join("data", "lancedb")
            shutil.rmtree(db_path, ignore_errors=True)
            os.makedirs(db_path, exist_ok=True)
            connect_lancedb()
            return jsonify({
                "response": "🧹 Database LanceDB cancellato e rigenerato da zero!"
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({"response": f"⚠️ Errore durante il reset: {str(e)}"})

    # === Chat normale ===
    try:
        response = ask_question(user_message)
        return jsonify({"response": response})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"response": f"⚠️ Errore durante l'elaborazione: {str(e)}"})
