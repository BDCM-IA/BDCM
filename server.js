const express = require("express");
const OpenAI = require("openai");

const app = express();

app.use(express.json());

/* Autoriser BDCM depuis GitHub Pages */
app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") {
        return res.sendStatus(200);
    }

    next();
});

/* Connexion à OpenAI */
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

/* Route BDCM */
app.post("/api/chat", async (req, res) => {
    try {
        const { message, history = [] } = req.body;

        if (!message) {
            return res.status(400).json({
                error: "Message manquant"
            });
        }

        const messages = [
            {
                role: "system",
                content:
                    "Tu es BDCM, un assistant IA personnel. Réponds en français, naturellement, clairement et de façon concise."
            },

            ...history.map(item => ({
                role: item.role,
                content: item.content
            })),

            {
                role: "user",
                content: message
            }
        ];

        const response = await openai.responses.create({
            model: "gpt-5-mini",
            input: messages
        });

        res.json({
            reply: response.output_text
        });

    } catch (error) {

        console.error("ERREUR OPENAI :", error);

        res.status(500).json({
            error: "Erreur avec OpenAI"
        });
    }
});

/* Port Render */
const PORT = process.env.PORT || 10000;

app.listen(PORT, "0.0.0.0", () => {
    console.log(`BDCM serveur lancé sur le port ${PORT}`);
});
