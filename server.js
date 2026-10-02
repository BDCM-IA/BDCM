const express = require("express");
const OpenAI = require("openai");

const app = express();

app.use(express.json());
app.use(express.static("."));

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

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
                content: "Tu es BDCM, un assistant IA personnel. Réponds en français, de façon naturelle, claire et concise."
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
        console.error(error);

        res.status(500).json({
            error: "Erreur avec l'API OpenAI"
        });
    }
});

const PORT = process.env.PORT || 10000;

app.listen(PORT, "0.0.0.0", () => {
    console.log(`BDCM serveur lancé sur le port ${PORT}`);
});
