const express = require("express");
const OpenAI = require("openai");
const Database = require("better-sqlite3");

const app = express();
const db = new Database("bdcm.db");

app.use(express.json({ limit: "100kb" }));
app.use(express.static("."));

/* =========================
   CORS
========================= */

app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") {
        return res.sendStatus(200);
    }

    next();
});

/* =========================
   DATABASE
========================= */

db.pragma("journal_mode = WAL");

db.exec(`
    CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        question_key TEXT NOT NULL UNIQUE,
        answer TEXT NOT NULL,
        source TEXT,
        embedding TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_used_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        hits INTEGER DEFAULT 0
    )
`);

/*
   Migration pour les anciennes bases
*/
const columns = db.prepare(`
    PRAGMA table_info(knowledge)
`).all();

const columnNames = new Set(columns.map(column => column.name));

if (!columnNames.has("question_key")) {
    db.exec(`
        ALTER TABLE knowledge
        ADD COLUMN question_key TEXT
    `);
}

if (!columnNames.has("embedding")) {
    db.exec(`
        ALTER TABLE knowledge
        ADD COLUMN embedding TEXT
    `);
}

if (!columnNames.has("last_used_at")) {
    db.exec(`
        ALTER TABLE knowledge
        ADD COLUMN last_used_at DATETIME DEFAULT CURRENT_TIMESTAMP
    `);
}

if (!columnNames.has("hits")) {
    db.exec(`
        ALTER TABLE knowledge
        ADD COLUMN hits INTEGER DEFAULT 0
    `);
}

/* =========================
   GROQ
========================= */

const openai = new OpenAI({
    apiKey: process.env.GROQ_API_KEY,
    baseURL: "https://api.groq.com/openai/v1"
});

const MODEL = "openai/gpt-oss-20b";

/* =========================
   EMBEDDINGS LOCAUX
========================= */

let embeddingPipeline = null;
let embeddingLoading = null;

async function getEmbeddingPipeline() {
    if (embeddingPipeline) {
        return embeddingPipeline;
    }

    if (embeddingLoading) {
        return embeddingLoading;
    }

    embeddingLoading = (async () => {
        const { pipeline } = await import("@huggingface/transformers");

        embeddingPipeline = await pipeline(
            "feature-extraction",
            "Xenova/all-MiniLM-L6-v2",
            {
                dtype: "q8"
            }
        );

        return embeddingPipeline;
    })();

    try {
        return await embeddingLoading;
    } finally {
        embeddingLoading = null;
    }
}

async function createEmbedding(text) {
    const extractor = await getEmbeddingPipeline();

    const output = await extractor(text, {
        pooling: "mean",
        normalize: true
    });

    return Array.from(output.data);
}

/* =========================
   NORMALISATION
========================= */

function normalizeText(text) {
    return text
        .toLowerCase()
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .replace(/[^\p{L}\p{N}\s]/gu, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function createQuestionKey(text) {
    const stopWords = new Set([
        "le", "la", "les", "un", "une", "des",
        "du", "de", "d", "et", "ou", "a", "à",
        "au", "aux", "en", "dans", "sur", "pour",
        "avec", "sans", "est", "sont", "je", "tu",
        "il", "elle", "on", "nous", "vous", "ils",
        "elles", "ce", "ça", "cela", "qui", "que",
        "quoi", "comment", "quel", "quelle", "quels",
        "quelles", "peut", "peux", "faire", "faire"
    ]);

    return normalizeText(text)
        .split(" ")
        .filter(word => word.length > 1 && !stopWords.has(word))
        .sort()
        .join(" ");
}

/* =========================
   COMPARAISON A
========================= */

function lexicalSimilarity(a, b) {
    const tokensA = new Set(createQuestionKey(a).split(" ").filter(Boolean));
    const tokensB = new Set(createQuestionKey(b).split(" ").filter(Boolean));

    if (!tokensA.size || !tokensB.size) {
        return 0;
    }

    let common = 0;

    for (const token of tokensA) {
        if (tokensB.has(token)) {
            common++;
        }
    }

    const union = new Set([...tokensA, ...tokensB]).size;

    return union === 0 ? 0 : common / union;
}

/* =========================
   COMPARAISON B
========================= */

function cosineSimilarity(a, b) {
    if (!a || !b || a.length !== b.length) {
        return 0;
    }

    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    if (!normA || !normB) {
        return 0;
    }

    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/* =========================
   CACHE
========================= */

const EXACT_CACHE = db.prepare(`
    SELECT id, answer
    FROM knowledge
    WHERE question_key = ?
    LIMIT 1
`);

const ALL_KNOWLEDGE = db.prepare(`
    SELECT id, question, answer, embedding
    FROM knowledge
    WHERE embedding IS NOT NULL
`);

const UPDATE_HIT = db.prepare(`
    UPDATE knowledge
    SET hits = hits + 1,
        last_used_at = CURRENT_TIMESTAMP
    WHERE id = ?
`);

const INSERT_KNOWLEDGE = db.prepare(`
    INSERT OR IGNORE INTO knowledge
    (
        question,
        question_key,
        answer,
        source,
        embedding
    )
    VALUES (?, ?, ?, ?, ?)
`);

async function findCachedAnswer(question) {
    const questionKey = createQuestionKey(question);

    /*
       A — correspondance exacte normalisée
    */

    const exact = EXACT_CACHE.get(questionKey);

    if (exact) {
        UPDATE_HIT.run(exact.id);

        return {
            answer: exact.answer,
            type: "exact"
        };
    }

    /*
       A — similarité lexicale
    */

    const candidates = ALL_KNOWLEDGE.all();

    let bestLexical = null;

    for (const item of candidates) {
        const score = lexicalSimilarity(question, item.question);

        if (
            score >= 0.82 &&
            (!bestLexical || score > bestLexical.score)
        ) {
            bestLexical = {
                ...item,
                score
            };
        }
    }

    if (bestLexical) {
        UPDATE_HIT.run(bestLexical.id);

        return {
            answer: bestLexical.answer,
            type: "lexical",
            score: bestLexical.score
        };
    }

    /*
       B — embedding sémantique
    */

    if (!candidates.length) {
        return null;
    }

    let currentEmbedding;

    try {
        currentEmbedding = await createEmbedding(question);
    } catch (error) {
        console.error(
            "Embedding indisponible :",
            error.message
        );

        return null;
    }

    let bestSemantic = null;

    for (const item of candidates) {
        if (!item.embedding) {
            continue;
        }

        try {
            const storedEmbedding = JSON.parse(item.embedding);

            const score = cosineSimilarity(
                currentEmbedding,
                storedEmbedding
            );

            if (
                score >= 0.86 &&
                (!bestSemantic || score > bestSemantic.score)
            ) {
                bestSemantic = {
                    ...item,
                    score
                };
            }
        } catch {
            // Embedding invalide : on l'ignore
        }
    }

    if (bestSemantic) {
        UPDATE_HIT.run(bestSemantic.id);

        return {
            answer: bestSemantic.answer,
            type: "semantic",
            score: bestSemantic.score
        };
    }

    return null;
}

/* =========================
   QUESTIONS DYNAMIQUES
========================= */

function needsWebSearch(message) {
    const text = normalizeText(message);

    const dynamicPatterns = [
        // Indications temporelles
        "aujourd hui",
        "maintenant",
        "en ce moment",
        "actuellement",
        "dernier",
        "derniere",
        "dernieres",
        "recent",
        "recente",
        "recemment",
        "news",
        "actualite",
        "actualites",
        "ce soir",
        "demain",
        "cette semaine",
        "cette annee",
        "2026",

        // Présent : être
        " est ",
        " sont ",

        // Présent : jouer
        " joue ",
        " jouent ",

        // Présent : avoir
        " a ",
        " ont ",

        // Présent : faire
        " fait ",
        " font ",

        // Présent : aller
        " va ",
        " vont ",

        // Présent : pouvoir
        " peut ",
        " peuvent ",

        // Présent : être dans / appartenir
        " se trouve ",
        " se trouvent ",
        " appartient ",
        " appartiennent ",

        const paddedText = ` ${text} `;
    return dynamicPatterns.some(pattern =>
        text.includes(pattern)
    );
}

/* =========================
   QUESTIONS PERSONNELLES
========================= */

function looksPersonal(message) {
    const text = normalizeText(message);

    const personalPatterns = [
        "mon adresse",
        "mon mot de passe",
        "mon compte",
        "mes donnees",
        "mes messages",
        "ma vie",
        "mon historique",
        "mes souvenirs",
        "je suis",
        "j ai",
        "j'aime",
        "jaime",
        "chez moi",
        "mon numero",
        "mon telephone"
    ];

    return personalPatterns.some(pattern =>
        text.includes(pattern)
    );
}

/* =========================
   PROMPT
========================= */

function buildSystemPrompt(mode, language) {
    let prompt = `
Tu es BDCM, un assistant IA personnel.

Réponds naturellement et clairement.
Réponds principalement en français sauf si la langue demandée est différente.

Ne fais pas de réponses inutilement longues.
Va directement à l'essentiel.

N'utilise pas de Markdown inutile lorsque la réponse doit être lue à voix haute.
Évite les listes avec trop de symboles.

Tu ne dois jamais prétendre connaître une information personnelle
sur l'utilisateur si elle ne t'a pas été fournie dans la conversation actuelle.
`;

    if (language === "en-US") {
        prompt += `
Réponds en anglais.
`;
    }

    if (language === "es-ES") {
        prompt += `
Réponds en espagnol.
`;
    }

    if (language === "de-DE") {
        prompt += `
Réponds en allemand.
`;
    }

    if (language === "it-IT") {
        prompt += `
Réponds en italien.
`;
    }

    if (mode === "agent") {
        prompt += `
Mode Agent :
sois particulièrement structuré et orienté vers l'action.
`;
    }

    if (mode === "ecoute") {
        prompt += `
Mode Écoute :
réponds de façon naturelle, courte et conversationnelle.
`;
    }

    if (mode === "chill") {
        prompt += `
Mode Chill :
adopte un ton détendu, naturel et amical.
`;
    }

    return prompt.trim();
}

/* =========================
   HISTORIQUE
========================= */

function cleanHistory(history) {
    if (!Array.isArray(history)) {
        return [];
    }

    return history
        .filter(item =>
            item &&
            ["user", "assistant"].includes(item.role) &&
            typeof item.content === "string"
        )
        .slice(-8)
        .map(item => ({
            role: item.role,
            content: item.content.slice(0, 4000)
        }));
}

/* =========================
   RATE LIMIT LOCAL
========================= */

const rateLimits = new Map();

function checkRateLimit(ip) {
    const now = Date.now();
    const windowMs = 60 * 1000;
    const maxRequests = 20;

    const data = rateLimits.get(ip);

    if (!data || now - data.start > windowMs) {
        rateLimits.set(ip, {
            start: now,
            count: 1
        });

        return true;
    }

    if (data.count >= maxRequests) {
        return false;
    }

    data.count++;

    return true;
}

/* =========================
   ROUTE CHAT
========================= */

app.post("/api/chat", async (req, res) => {
    try {
        if (!process.env.GROQ_API_KEY) {
            return res.status(500).json({
                error: "Clé Groq non configurée sur le serveur."
            });
        }

        const ip = req.ip || "unknown";

        if (!checkRateLimit(ip)) {
            return res.status(429).json({
                error: "Trop de requêtes. Réessaie dans quelques secondes."
            });
        }

        const {
            message,
            history = [],
            mode = "chill",
            language = "fr-FR"
        } = req.body;

        if (
            typeof message !== "string" ||
            !message.trim()
        ) {
            return res.status(400).json({
                error: "Message manquant."
            });
        }

        const cleanMessage = message.trim();

        /*
           Questions personnelles :
           pas de cache commun.
        */

        const personal = looksPersonal(cleanMessage);

        /*
           Questions dynamiques :
           pas de cache commun pour éviter
           les réponses périmées.
        */

        const dynamic = needsWebSearch(cleanMessage);

        /*
           CACHE COMMUN
        */

        if (!personal && !dynamic) {
            const cached = await findCachedAnswer(
                cleanMessage
            );

            if (cached) {
                return res.json({
                    reply: cached.answer,
                    cached: true,
                    cacheType: cached.type
                });
            }
        }

        /*
           CONSTRUCTION DES MESSAGES
        */

        const messages = [
            {
                role: "system",
                content: buildSystemPrompt(
                    mode,
                    language
                )
            },

            ...cleanHistory(history),

            {
                role: "user",
                content: cleanMessage
            }
        ];

        /*
           GROQ
        */

        const requestOptions = {
            model: MODEL,
            input: messages
        };

        /*
           Recherche web uniquement
           quand la question semble dynamique.
        */

        if (dynamic) {
            requestOptions.tools = [
                {
                    type: "browser_search"
                }
            ];

            requestOptions.tool_choice = "required";
        }

        const response =
            await openai.responses.create(
                requestOptions
            );

        const answer =
            response.output_text?.trim();

        if (!answer) {
            return res.status(500).json({
                error: "BDCM n'a pas produit de réponse."
            });
        }

        /*
           STOCKAGE DU SAVOIR COMMUN
        */

        if (!personal && !dynamic) {
            try {
                const embedding =
                    await createEmbedding(
                        cleanMessage
                    );

                INSERT_KNOWLEDGE.run(
                    cleanMessage,
                    createQuestionKey(
                        cleanMessage
                    ),
                    answer,
                    null,
                    JSON.stringify(embedding)
                );
            } catch (error) {
                /*
                   Même si l'embedding échoue,
                   la réponse reste utilisable.
                */

                console.error(
                    "Impossible de sauvegarder l'embedding :",
                    error.message
                );

                INSERT_KNOWLEDGE.run(
                    cleanMessage,
                    createQuestionKey(
                        cleanMessage
                    ),
                    answer,
                    null,
                    null
                );
            }
        }

        return res.json({
            reply: answer,
            cached: false
        });

    } catch (error) {
        console.error(
            "ERREUR BDCM :",
            error
        );

        return res.status(500).json({
            error: "Erreur avec BDCM."
        });
    }
});

/* =========================
   HEALTH CHECK
========================= */

app.get("/api/health", (req, res) => {
    res.json({
        status: "ok",
        service: "BDCM"
    });
});

/* =========================
   PORT RENDER
========================= */

const PORT =
    process.env.PORT || 10000;

app.listen(
    PORT,
    "0.0.0.0",
    () => {
        console.log(
            `BDCM serveur lancé sur le port ${PORT}`
        );
    }
);
