/**
 * cardio.routes.js
 * GET  /cardio/patients        – list all rows from final_data + RF prediction
 * GET  /cardio/patient/:id     – single row prediction + Gemini analysis
 * GET  /cardio/latest          – most recent row with full Gemini analysis
 */

const express  = require("express");
const router   = express.Router();
// Uses Node 18+ built-in fetch — no extra package needed
const { createClient } = require("@supabase/supabase-js");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_KEY
);

const genAI  = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const ML_URL = process.env.ML_SERVICE_URL || "http://localhost:5001";

/* ── clinical defaults for columns absent from IoT table ───────────────── */
const CLINICAL_DEFAULTS = {
  fbs:         0,
  thalach:     150,
  slope:       1,
  ca:          0,
  thal:        2,
  spo2:        97.5,
  temperature: 37.0,
};

/**
 * Merge a final_data row with clinical defaults.
 * NULL IoT fields also fall back to safe defaults.
 */
function buildPayload(row) {
  return {
    age:         row.age         ?? 50,
    sex:         row.sex         ?? 1,
    cp:          row.cp          ?? 0,
    trestbps:    row.trestbps    ?? 120,
    chol:        row.chol        ?? 200,
    fbs:         row.fbs         ?? CLINICAL_DEFAULTS.fbs,
    restecg:     row.restecg     ?? 0,
    thalach:     row.thalach     ?? CLINICAL_DEFAULTS.thalach,
    exang:       row.exang       ?? 0,
    oldpeak:     row.oldpeak     ?? 0,
    slope:       row.slope       ?? CLINICAL_DEFAULTS.slope,
    ca:          row.ca          ?? CLINICAL_DEFAULTS.ca,
    thal:        row.thal        ?? CLINICAL_DEFAULTS.thal,
    spo2:        row.spo2        ?? CLINICAL_DEFAULTS.spo2,
    temperature: row.temperature ?? CLINICAL_DEFAULTS.temperature,
  };
}

/** Call ML microservice */
async function mlPredict(payload) {
  const r = await fetch(`${ML_URL}/predict`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(`ML service error: ${r.status}`);
  return r.json();
}

/** Gemini clinical interpretation (multi-model fallback) */
const GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.6-flash"];

async function geminiAnalyze(row, ml) {
  const flagList     = (ml.flagged_features || []).map(f => `${f.label}: ${f.value}`).join(", ") || "None";
  const diseaseList  = (ml.disease_breakdown || []).map(d => `${d.name} (${d.pct}%)`).join(", ");
  const sexLabel     = row.sex === 1 ? "Male" : "Female";

  const prompt = `
You are a senior cardiologist AI. A patient record was pulled from an IoT health monitoring system.

PATIENT:
  Age: ${row.age}, Sex: ${sexLabel}
  BP: ${row.trestbps} mmHg, Cholesterol: ${row.chol} mg/dL
  Chest Pain Type: ${row.cp}, Exercise Angina: ${row.exang === 1 ? "Yes" : "No"}
  ST Depression: ${row.oldpeak ?? "N/A"}, Resting ECG: ${row.restecg ?? "N/A"}
  Record date: ${row.created_at}

ML ASSESSMENT:
  Overall Risk: ${ml.risk_percentage}% (${ml.risk_level})
  Disease Breakdown: ${diseaseList}
  Abnormal Readings: ${flagList}

Return ONLY this JSON (no markdown):
{
  "summary": "2-3 sentence plain-language assessment",
  "primaryConcern": "single most critical finding",
  "recommendations": ["3-4 actionable clinical steps"],
  "lifestyle": ["2-3 daily habit changes"],
  "urgency": "Routine | Soon | Urgent | Emergency",
  "disclaimer": "one-line medical disclaimer"
}`.trim();

  for (const modelName of GEMINI_MODELS) {
    try {
      console.log(`[cardio.routes] Trying Gemini: ${modelName}`);
      const model   = genAI.getGenerativeModel({ model: modelName });
      const result  = await model.generateContent(prompt);
      const rawText = result.response.text().trim();

      try {
        const parsed = JSON.parse(rawText);
        console.log(`[cardio.routes] ✓ Success with ${modelName}`);
        return parsed;
      } catch {
        const clean = rawText.replace(/```json|```/gi, "").trim();
        try {
          const parsed = JSON.parse(clean);
          console.log(`[cardio.routes] ✓ Success with ${modelName}`);
          return parsed;
        } catch {
          console.log(`[cardio.routes] ✓ ${modelName} returned non-JSON, using raw text`);
          return { summary: rawText };
        }
      }
    } catch (err) {
      const msg = err.message || "";
      if (msg.includes("404")) {
        console.warn(`[cardio.routes] ${modelName} → 404 deprecated, skipping`);
        continue;
      }
      if (msg.includes("429")) {
        console.warn(`[cardio.routes] ${modelName} → 429 rate-limited, trying next`);
        continue;
      }
      console.warn(`[cardio.routes] ${modelName} failed: ${msg}`);
    }
  }

  throw new Error("All Gemini models failed");
}

/* ── GET /cardio/patients ─────────────────────────────────────────────── */
router.get("/patients", async (req, res) => {
  try {
    const userId = req.query.user_id;
    if (!userId) return res.status(401).json({ error: "user_id is required" });

    const { data: rows, error } = await supabase
      .from("final_data")
      .select("*")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(50);

    if (error) return res.status(500).json({ error: error.message });

    // Run ML predictions in parallel (no Gemini here – just scores)
    const predictions = await Promise.all(
      rows.map(async (row) => {
        try {
          const payload = buildPayload(row);
          const ml      = await mlPredict(payload);
          return {
            id:           row.id,
            created_at:   row.created_at,
            age:          row.age,
            sex:          row.sex,
            trestbps:     row.trestbps,
            chol:         row.chol,
            risk_percentage:   ml.risk_percentage,
            risk_level:        ml.risk_level,
            disease_breakdown: ml.disease_breakdown,
            flagged_count:     (ml.flagged_features || []).length,
          };
        } catch {
          return { id: row.id, created_at: row.created_at, error: true };
        }
      })
    );

    res.json({ count: predictions.length, patients: predictions });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ── GET /cardio/patient/:id ──────────────────────────────────────────── */
router.get("/patient/:id", async (req, res) => {
  try {
    const userId = req.query.user_id;
    if (!userId) return res.status(401).json({ error: "user_id is required" });

    const { data: rows, error } = await supabase
      .from("final_data")
      .select("*")
      .eq("id", req.params.id)
      .eq("user_id", userId)
      .limit(1);

    if (error || !rows.length)
      return res.status(404).json({ error: "Patient not found" });

    const row     = rows[0];
    const payload = buildPayload(row);
    const ml      = await mlPredict(payload);

    // ── Gemini is optional — never crash the whole response if it fails ──
    let gemini      = null;
    let geminiError = null;
    try {
      gemini = await geminiAnalyze(row, ml);
    } catch (gErr) {
      geminiError = gErr.message || "Gemini unavailable";
    }

    res.json({ row, payload_used: payload, ...ml, gemini, geminiError });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ── GET /cardio/latest ────────────────────────────────────────────────── */
router.get("/latest", async (req, res) => {
  try {
    const userId = req.query.user_id;
    if (!userId) return res.status(401).json({ error: "user_id is required" });

    const { data: rows, error } = await supabase
      .from("final_data")
      .select("*")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(1);

    if (error || !rows.length)
      return res.status(404).json({ error: "No records found" });

    const row     = rows[0];
    const payload = buildPayload(row);
    const ml      = await mlPredict(payload);

    // ── Gemini is optional — never crash the whole response if it fails ──
    let gemini      = null;
    let geminiError = null;
    try {
      gemini = await geminiAnalyze(row, ml);
    } catch (gErr) {
      geminiError = gErr.message || "Gemini unavailable";
    }

    res.json({ row, payload_used: payload, ...ml, gemini, geminiError });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;