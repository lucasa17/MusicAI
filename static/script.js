
const API_URL = "";

// Features na mesma ordem usada pelo modelo durante o treinamento
const FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness"
];

// Descrições dos clusters
const CLUSTER_INFO = {
    0: {
        icon: "🔥",
        description: "Uma música de alta energia, perfeita para festas, pista de dança e momentos intensos."
    },

    1: {
        icon: "🌿",
        description: "Uma vibe acústica e tranquila. Sons suaves para relaxar e desacelerar."
    },

    2: {
        icon: "🎧",
        description: "Dance e eletrônica moderna. Batidas digitais e uma atmosfera vibrante."
    },

    3: {
        icon: "🌙",
        description: "Uma atmosfera melancólica, introspectiva e contemplativa, com uma vibe lo-fi."
    },

    4: {
        icon: "🎤",
        description: "Uma identidade urbana, com ritmo marcante e presença vocal."
    }
};

// Elementos da interface
const form = document.getElementById("musicForm");

const button = document.getElementById("classifyButton");

const resultSection = document.getElementById("resultSection");

const errorMessage = document.getElementById("errorMessage");


// ====================================
// ATUALIZAÇÃO DOS SLIDERS
// ====================================

FEATURES.forEach((feature) => {

    const slider = document.getElementById(feature);

    const output = document.getElementById(
        `${feature}Value`
    );

    slider.addEventListener("input", () => {

        const value = Number(slider.value);

        if (feature === "tempo") {
            output.textContent = `${value} BPM`;
        } else {
            output.textContent = value.toFixed(2);
        }

    });

});


// ====================================
// VERIFICAR CONEXÃO COM A API
// ====================================

async function checkAPI() {

    const indicator = document.getElementById(
        "statusIndicator"
    );

    const statusText = document.getElementById(
        "statusText"
    );

    try {

        const response = await fetch(API_URL + "/api");

        if (!response.ok) {
            throw new Error("API indisponível");
        }

        const data = await response.json();

        indicator.classList.remove("offline");
        indicator.classList.add("online");

        statusText.textContent = data.status;

    } catch (error) {

        indicator.classList.remove("online");
        indicator.classList.add("offline");

        statusText.textContent = "API Offline";

    }

}


// ====================================
// COLETAR DADOS DOS SLIDERS
// ====================================

function collectFeatures() {

    const dados = {};

    FEATURES.forEach((feature) => {

        const slider = document.getElementById(feature);

        dados[feature] = Number(slider.value);

    });

    return dados;

}


// ====================================
// CLASSIFICAR MÚSICA
// ====================================

form.addEventListener("submit", async (event) => {

    event.preventDefault();

    errorMessage.classList.add("hidden");

    resultSection.classList.add("hidden");

    button.disabled = true;

    button.innerHTML = `
        <span>⏳</span>
        Classificando música...
    `;

    try {

        const dados = collectFeatures();

        const response = await fetch(
            `${API_URL}/api/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(dados)
            }
        );

        const resultado = await response.json();

        if (!response.ok || !resultado.sucesso) {

            throw new Error(
                resultado.erro || "Erro ao classificar música."
            );

        }

        // Exibir resultado
        displayResult(resultado);

    } catch (error) {

        errorMessage.textContent =
            `Erro: ${error.message}. Verifique se a API Flask está rodando na porta 5001.`;

        errorMessage.classList.remove("hidden");

    } finally {

        button.disabled = false;

        button.innerHTML = `
            <span>♫</span>
            Classificar minha música
            <span class="button-arrow">→</span>
        `;

    }

});


// ====================================
// EXIBIR RESULTADO
// ====================================

function displayResult(resultado) {

    const clusterId = resultado.cluster_id;

    const vibe = resultado.vibe;

    const info = CLUSTER_INFO[clusterId];

    document.getElementById("clusterBadge").textContent =
        `CLUSTER ${clusterId}`;

    document.getElementById("resultIcon").textContent =
        info ? info.icon : "🎵";

    document.getElementById("vibeName").textContent =
        vibe;

    document.getElementById("resultDescription").textContent =
        info
            ? info.description
            : "A música foi classificada pelo modelo.";

    document.getElementById("resultTimestamp").textContent =
        new Date().toLocaleTimeString("pt-BR");

    resultSection.classList.remove("hidden");

    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });

}


// ====================================
// INICIALIZAÇÃO
// ====================================

checkAPI();
