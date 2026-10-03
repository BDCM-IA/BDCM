const express = require("express");
const OpenAI = require("openai");
const Database = require("better-sqlite3");

const db = new Database("bdcm.db");

db.exec(`
    CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        source TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
`);
const app = express();

app.use(express.json());
app.use(express.static("."));

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

/* Connexion à Groq */
const openai = new OpenAI({
    apiKey: process.env.GROQ_API_KEY,
    baseURL: "https://api.groq.com/openai/v1"
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
const cached = db.prepare(`
    SELECT answer
    FROM knowledge
    WHERE question = ?
    LIMIT 1
`).get(message);

if (cached) {
    return res.json({
        reply: cached.answer
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
    model: "openai/gpt-oss-20b",
    input: messages,
    tools: [
        {
            type: "browser_search"
        }
    ]
});

        const answer = response.output_text;

db.prepare(`
    INSERT INTO knowledge (question, answer)
    VALUES (?, ?)
`).run(message, answer);

res.json({
    reply: answer
});

    } catch (error) {
        console.error("ERREUR GROQ :", error);

        res.status(500).json({
            error: "Erreur avec Groq"
        });
    }
});

/* Port Render */
const PORT = process.env.PORT || 10000;

app.listen(PORT, "0.0.0.0", () => {
    console.log(`BDCM serveur lancé sur le port ${PORT}`);
});
