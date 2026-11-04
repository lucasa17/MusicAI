import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import json
import plotly
import plotly.express as px
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Carregar o modelo e o scaler treinados
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Dicionário descritivo para as 5 "vibes"
cluster_labels = {
    0: "🔥 Alta Energia / Festa e Pista",
    1: "🌿 Acústica & Calma (Relax)",
    2: "🎧 Dance & Eletrônica Moderna",
    3: "🌙 Melancólica / Lo-fi",
    4: "🎤 Urbano / Ritmo Marcante"
}

# Dicionário para traduzir gêneros musicais para PT-BR
TRADUCAO_GENEROS = {
    "acoustic": "Acústico",
    "afrobeat": "Afrobeat",
    "alt-rock": "Rock Alternativo",
    "alternative": "Alternativo",
    "ambient": "Ambiente",
    "anime": "Anime",
    "black-metal": "Black Metal",
    "bluegrass": "Bluegrass",
    "blues": "Blues",
    "brazil": "Música Brasileira",
    "breakbeat": "Breakbeat",
    "british": "Britânico",
    "cantopop": "Cantopop",
    "chicago-house": "Chicago House",
    "children": "Infantil",
    "chill": "Chill / Relax",
    "classical": "Clássica",
    "club": "Club / Balada",
    "comedy": "Comédia",
    "country": "Country",
    "dance": "Dance",
    "dancehall": "Dancehall",
    "death-metal": "Death Metal",
    "deep-house": "Deep House",
    "detroit-techno": "Detroit Techno",
    "disco": "Disco",
    "disney": "Disney",
    "drum-and-bass": "Drum and Bass",
    "dub": "Dub",
    "dubstep": "Dubstep",
    "edm": "EDM / Eletrônica",
    "electro": "Electro",
    "electronic": "Eletrônica",
    "emo": "Emo",
    "folk": "Folk",
    "forro": "Forró",
    "french": "Francês",
    "funk": "Funk",
    "garage": "Garage",
    "german": "Alemão",
    "gospel": "Gospel",
    "goth": "Gótico",
    "grindcore": "Grindcore",
    "groove": "Groove",
    "grunge": "Grunge",
    "guitar": "Violão / Guitarra",
    "happy": "Alegre",
    "hard-rock": "Hard Rock",
    "hardcore": "Hardcore",
    "hardstyle": "Hardstyle",
    "heavy-metal": "Heavy Metal",
    "hip-hop": "Hip-Hop",
    "honky-tonk": "Honky-Tonk",
    "house": "House",
    "idm": "IDM",
    "indian": "Indiano",
    "indie": "Indie",
    "indie-pop": "Indie Pop",
    "industrial": "Industrial",
    "iranian": "Iraniano",
    "j-dance": "J-Dance",
    "j-idol": "J-Idol",
    "j-pop": "J-Pop",
    "j-rock": "J-Rock",
    "jazz": "Jazz",
    "k-pop": "K-Pop",
    "kids": "Infantil",
    "latin": "Latina",
    "latino": "Latino",
    "malay": "Malaio",
    "mandopop": "Mandopop",
    "metal": "Metal",
    "metalcore": "Metalcore",
    "minimal-techno": "Techno Minimalista",
    "mpb": "MPB",
    "new-age": "New Age",
    "opera": "Ópera",
    "pagode": "Pagode",
    "party": "Festa",
    "piano": "Piano",
    "pop": "Pop",
    "pop-film": "Pop Trilha Sonora",
    "post-dubstep": "Post-Dubstep",
    "power-pop": "Power Pop",
    "progressive-house": "Progressive House",
    "psych-rock": "Rock Psicodélico",
    "punk": "Punk",
    "punk-rock": "Punk Rock",
    "r-n-b": "R&B",
    "reggae": "Reggae",
    "reggaeton": "Reggaeton",
    "rock": "Rock",
    "rock-n-roll": "Rock and Roll",
    "rockabilly": "Rockabilly",
    "romance": "Romântica",
    "sad": "Triste",
    "salsa": "Salsa",
    "samba": "Samba",
    "sertanejo": "Sertanejo",
    "show-tunes": "Trilhas de Teatro",
    "singer-songwriter": "Cantor-Compositor",
    "ska": "Ska",
    "sleep": "Sono / Relax",
    "soul": "Soul",
    "spanish": "Espanhol",
    "study": "Estudo",
    "swedish": "Sueco",
    "synth-pop": "Synth-pop",
    "tango": "Tango",
    "techno": "Techno",
    "trance": "Trance",
    "trip-hop": "Trip-Hop",
    "turkish": "Turco",
    "world-music": "World Music"
}

# Cache do dataset para o dashboard
DF_CACHE = None

def get_dataset():
    global DF_CACHE
    if DF_CACHE is None:
        csv_file = Path("dataset.csv")
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            if "" in df.columns or "Unnamed: 0" in df.columns:
                col_unnamed = "" if "" in df.columns else "Unnamed: 0"
                df = df.drop(columns=[col_unnamed], errors="ignore")
            # Traduzir a coluna de gêneros no dataset
            if "track_genre" in df.columns:
                df["track_genre"] = df["track_genre"].map(lambda g: TRADUCAO_GENEROS.get(str(g).lower(), str(g).capitalize()))
            DF_CACHE = df
        else:
            DF_CACHE = pd.DataFrame()
    return DF_CACHE

def criar_dataset_amigavel(df_orig: pd.DataFrame, n_linhas: int = 20) -> pd.DataFrame:
    """
    Recria o dataset selecionando e renomeando as colunas conforme o modelo do usuário:
    pd.DataFrame({ 'Nome': [...], 'Idade': [...], ... })
    """
    if df_orig.empty:
        return pd.DataFrame()

    sub = df_orig.head(n_linhas).copy()

    # Formatar duração: de milissegundos para MM:SS min
    def formatar_duracao(ms):
        try:
            total_s = int(ms) // 1000
            m = total_s // 60
            s = total_s % 60
            return f"{m}:{s:02d} min"
        except:
            return ms

    # Arredondar features numéricas
    def arr(val):
        try:
            return f"{round(float(val), 2):.2f}"
        except:
            return "-"

    # Recriar explicitamente o DataFrame com as colunas em Português
    df_amigavel = pd.DataFrame({
        "Artista": sub["artists"] if "artists" in sub.columns else [],
        "Nome da Música": sub["track_name"] if "track_name" in sub.columns else [],
        "Álbum": sub["album_name"] if "album_name" in sub.columns else [],
        "Gênero": sub["track_genre"] if "track_genre" in sub.columns else [],
        "Popularidade": sub["popularity"].astype(int) if "popularity" in sub.columns else [],
        "Duração": sub["duration_ms"].apply(formatar_duracao) if "duration_ms" in sub.columns else [],
        "Explícito": sub["explicit"].apply(lambda v: "Sim" if str(v).lower() in ["true", "1"] else "Não") if "explicit" in sub.columns else [],
        "Dançabilidade": sub["danceability"].apply(arr) if "danceability" in sub.columns else [],
        "Energia": sub["energy"].apply(arr) if "energy" in sub.columns else [],
        "Acústica": sub["acousticness"].apply(arr) if "acousticness" in sub.columns else [],
        "Instrumental": sub["instrumentalness"].apply(arr) if "instrumentalness" in sub.columns else [],
        "Presença Vocal": sub["speechiness"].apply(arr) if "speechiness" in sub.columns else [],
        "Ao Vivo": sub["liveness"].apply(arr) if "liveness" in sub.columns else [],
        "Positividade": sub["valence"].apply(arr) if "valence" in sub.columns else [],
        "Ritmo (BPM)": sub["tempo"].apply(lambda v: f"{round(float(v), 0):.0f}" if pd.notnull(v) else "-") if "tempo" in sub.columns else [],
        "Modo": sub["mode"].apply(lambda v: "Maior" if str(v) == "1" else "Menor") if "mode" in sub.columns else []
    })

    return df_amigavel

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/dashboard", methods=["GET"])
def dashboard():
    df = get_dataset()
    
    # Recria o dataset com colunas amigáveis em Português
    df_amigavel = criar_dataset_amigavel(df, n_linhas=20)
    table_sample = df_amigavel.to_dict(orient="records") if not df_amigavel.empty else []
    table_columns = list(df_amigavel.columns) if not df_amigavel.empty else []
    
    # Métricas / KPIs
    total_tracks = int(len(df))
    avg_danceability = round(float(df["danceability"].mean()), 2) if "danceability" in df.columns and not df.empty else 0
    avg_energy = round(float(df["energy"].mean()), 2) if "energy" in df.columns and not df.empty else 0
    top_genre = df["track_genre"].mode().iat[0] if "track_genre" in df.columns and not df["track_genre"].dropna().empty else "N/D"

    plotly_template = "plotly_dark"

    # 1. Top Gêneros mais frequentes
    chart_genres_json = "{}"
    if "track_genre" in df.columns and not df.empty:
        genre_counts = df["track_genre"].value_counts().head(10).reset_index()
        genre_counts.columns = ["Gênero", "Quantidade"]
        fig_genres = px.bar(
            genre_counts,
            x="Quantidade",
            y="Gênero",
            orientation="h",
            title="Top 10 Gêneros Mais Frequentes",
            template=plotly_template,
            color="Quantidade",
            color_continuous_scale="Purples"
        )
        fig_genres.update_layout(
            paper_bgcolor="rgba(18, 18, 28, 0.8)",
            plot_bgcolor="rgba(18, 18, 28, 0.8)",
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(autorange="reversed")
        )
        chart_genres_json = json.dumps(fig_genres, cls=plotly.utils.PlotlyJSONEncoder)

    # 2. Distribuição de Popularidade
    chart_pop_json = "{}"
    if "popularity" in df.columns and not df.empty:
        fig_pop = px.histogram(
            df.head(2000),
            x="popularity",
            nbins=30,
            title="Distribuição da Popularidade",
            labels={"popularity": "Nível de Popularidade (0 a 100)"},
            template=plotly_template,
            color_discrete_sequence=["#a855f7"]
        )
        fig_pop.update_layout(
            paper_bgcolor="rgba(18, 18, 28, 0.8)",
            plot_bgcolor="rgba(18, 18, 28, 0.8)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        chart_pop_json = json.dumps(fig_pop, cls=plotly.utils.PlotlyJSONEncoder)

    # 3. Dispersão: Dançabilidade vs Energia
    chart_scatter_json = "{}"
    if {"danceability", "energy"}.issubset(df.columns) and not df.empty:
        sample_scatter = df.head(500)
        fig_scatter = px.scatter(
            sample_scatter,
            x="danceability",
            y="energy",
            color="track_genre" if "track_genre" in sample_scatter.columns else None,
            title="Dançabilidade vs Energia das Músicas",
            labels={
                "danceability": "Dançabilidade (0 a 1)",
                "energy": "Energia (0 a 1)",
                "track_genre": "Gênero"
            },
            template=plotly_template,
            opacity=0.75
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(18, 18, 28, 0.8)",
            plot_bgcolor="rgba(18, 18, 28, 0.8)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        chart_scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "dashboard.html",
        table_sample=table_sample,
        table_columns=table_columns,
        total_tracks=total_tracks,
        avg_danceability=avg_danceability,
        avg_energy=avg_energy,
        top_genre=top_genre,
        chart_genres_json=chart_genres_json,
        chart_pop_json=chart_pop_json,
        chart_scatter_json=chart_scatter_json
    )

# Rota API opcional para retornar o dataset tratado em JSON
@app.route("/api/dataset", methods=["GET"])
def api_dataset():
    df = get_dataset()
    df_amigavel = criar_dataset_amigavel(df, n_linhas=20)
    return jsonify(df_amigavel.to_dict(orient="records"))

@app.route("/api", methods=["GET"])
def api():
    return jsonify({"status": "API Online", "message": "API do K-Means de Músicas rodando!"})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        dados = request.get_json(force=True)
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
