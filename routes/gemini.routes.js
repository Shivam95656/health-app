
const express     = require("express");
const router      = express.Router();
const buildPrompt = require("../prompts/heartAnalysis.prompt");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const GEMINI_KEY = process.env.GEMINI_API_KEY || "";
const genAI      = GEMINI_KEY ? new GoogleGenerativeAI(GEMINI_KEY) : null;
const GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.6-flash"];

// POST /gemini
router.post("/", async (req, res) => {
  const { profile, iot, risk, breakdown } = req.body;

  if (!profile || !iot || risk === undefined) {
    return res.status(400).json({ error: "profile, iot, and risk are required." });
  }

  if (!genAI) {
    return res.status(503).json({ error: "Gemini API key not configured." });
  }

  const prompt = buildPrompt(profile, iot, risk, breakdown);

  for (const modelName of GEMINI_MODELS) {
    try {
      console.log(`[gemini.routes] Trying: ${modelName}`);
      const model  = genAI.getGenerativeModel({ model: modelName });
      const result = await model.generateContent(prompt);
      const text   = result.response.text().trim() || "No response from Gemini.";

      console.log(`[gemini.routes] ✓ Success with ${modelName}`);
      return res.json({ text });
    } catch (err) {
      const msg = err.message || "";
      if (msg.includes("404")) {
        console.warn(`[gemini.routes] ${modelName} → 404 deprecated, skipping`);
        continue;
      }
      if (msg.includes("429")) {
        console.warn(`[gemini.routes] ${modelName} → 429 rate-limited, trying next`);
        continue;
      }
      console.warn(`[gemini.routes] ${modelName} failed: ${msg}`);
    }
  }

  res.status(502).json({ error: "Gemini API error: all models failed." });
});

module.exports = router;



