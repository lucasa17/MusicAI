import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Carregar o modelo e o scaler treinados
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Dicionário descritivo para as 5 "vibes"
cluster_labels = {
    0: "🔥 High Energy / Festa e Pista",
    1: "🌿 Acústica & Calma (Relax)",
    2: "🎧 Dance & Eletrônica Moderna",
    3: "🌙 Melancólica / Lo-fi",
    4: "🎤 Urbano / Ritmo Marcante"
}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/api", methods=["GET"])
def api():
    return jsonify({"status": "API Online", "message": "API do K-Means de Músicas rodando!"})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        dados = request.get_json(force=True)
        # Espera receber um JSON com as features: danceability, energy, valence, tempo, acousticness, instrumentalness, speechiness
        valores_entrada = list(dados.values())
        array_features = np.array([valores_entrada])
        
        # Normalizar os dados de entrada usando o mesmo scaler do treino
        array_scaled = scaler.transform(array_features)
        
        # Prever o cluster
        cluster_id = int(model.predict(array_scaled)[0])
        vibe_nome = cluster_labels.get(cluster_id, "Desconhecida")
        
        return jsonify({
            "sucesso": True,
            "cluster_id": cluster_id,
            "vibe": vibe_nome,
            "dados_recebidos": dados
        })
    except Exception as e:
        return jsonify({"sucesso": False, "erro": str(e)}), 400

if __name__ == "__main__":
    app.run(port=5001, debug=True)