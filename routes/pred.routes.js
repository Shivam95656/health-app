const express = require("express");
const router  = express.Router();
// Uses Node 18+ built-in fetch — no extra package needed
const { GoogleGenerativeAI } = require("@google/generative-ai");

const ML_URL = process.env.ML_SERVICE_URL || "http://localhost:5001";
const genAI  = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// ── POST /pred/analyze ────────────────────────────────────────────────────
router.post("/analyze", async (req, res) => {
  try {
    // 1. Call Python RF service
    const mlRes = await fetch(`${ML_URL}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(req.body),
    });

    if (!mlRes.ok) {
      const err = await mlRes.text();
      return res.status(502).json({ error: "ML service error", detail: err });
    }

    const ml = await mlRes.json();

    // 2. Build Gemini prompt
    const flagList = ml.flagged_features
      .map(f => `${f.label}: ${f.value}`)
      .join(", ") || "None";

    const diseaseList = ml.disease_breakdown
      .map(d => `${d.name} (${d.pct}%)`)
      .join(", ");

    const prompt = `
You are a senior cardiologist AI assistant. A patient's heart-disease risk has been
evaluated by a machine-learning model.

PATIENT INPUT:
  Age: ${req.body.age}, Sex: ${req.body.sex === "1" || req.body.sex === 1 ? "Male" : "Female"}
  BP: ${req.body.trestbps} mmHg, Cholesterol: ${req.body.chol} mg/dL
  Max Heart Rate: ${req.body.thalach} bpm, SpO₂: ${req.body.spo2}%
  Temperature: ${req.body.temperature}°C, Chest Pain Type: ${req.body.cp}
  Exercise Angina: ${req.body.exang === "1" || req.body.exang === 1 ? "Yes" : "No"}
  ST Depression (Oldpeak): ${req.body.oldpeak}

ML RESULTS:
  Overall Heart Disease Risk: ${ml.risk_percentage}% (${ml.risk_level})
  Disease Breakdown: ${diseaseList}
  Abnormal Readings: ${flagList}

Provide a structured clinical interpretation with these exact JSON keys:
{
  "summary": "2–3 sentence overall assessment in plain language",
  "primaryConcern": "the single most critical finding",
  "recommendations": ["array", "of", "3–4 actionable", "lifestyle or medical steps"],
  "lifestyle": ["array", "of", "2–3 specific daily habits to adopt"],
  "urgency": "Routine | Soon | Urgent | Emergency",
  "disclaimer": "one-line medical disclaimer"
}

Respond with ONLY valid JSON, no markdown fences, no preamble.
`.trim();

    const geminiModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    const gemResult   = await geminiModel.generateContent(prompt);
    const rawText     = gemResult.response.text().trim();

    let gemini = {};
    try {
      gemini = JSON.parse(rawText);
    } catch {
      // Fallback if Gemini wraps in ```json
      const clean = rawText.replace(/```json|```/gi, "").trim();
      try { gemini = JSON.parse(clean); } catch { gemini = { summary: rawText }; }
    }

    return res.json({ ...ml, gemini });

  } catch (err) {
    console.error("[pred.routes] Error:", err.message);
    res.status(500).json({ error: "Prediction failed", detail: err.message });
  }
});

module.exports = router;
