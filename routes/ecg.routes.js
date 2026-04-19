/**
 * ecg.routes.js
 * Fetches raw ECG waveform data from Supabase and runs heart-disease prediction.
 *
 * Assumed Supabase table  →  ecg_data
 *   id           uuid  PK
 *   user_id      uuid  FK → auth.users
 *   created_at   timestamptz
 *   samples      jsonb   -- float[]  raw ADC / mV values
 *   sample_rate  int     -- Hz (e.g. 360)
 *   heart_rate   float   -- bpm derived on device / CNN service
 *   label        text    -- CNN arrhythmia label  e.g. "Normal" | "AFIB" | "ST-elevation"
 *   duration_sec float
 *
 * Routes
 *   GET  /ecg/latest/:user_id   – latest record with full samples
 *   GET  /ecg/history/:user_id  – last 10 records (no samples, for list)
 *   GET  /ecg/record/:id        – single record by id with full samples
 *   POST /ecg/predict/:user_id  – latest ECG + profile → ML + Gemini
 */

const express  = require("express");
const router   = express.Router();
const supabase = require("../supabase");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const ML_URL = process.env.ML_SERVICE_URL || "http://localhost:5001";
const genAI  = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

/* ─── helpers ─────────────────────────────────────────────────────────── */

/**
 * Map CNN arrhythmia label  →  restecg integer (0 | 1 | 2)
 * 0 = Normal
 * 1 = ST-T wave abnormality (AFIB, ST-elevation/depression, T-wave inversion)
 * 2 = Left ventricular hypertrophy / other severe
 */
function labelToRestecg(label = "") {
  const l = label.toLowerCase();
  if (!l || l === "normal" || l === "n") return 0;
  if (l.includes("afib") || l.includes("st") || l.includes("svt") ||
      l.includes("pvc")  || l.includes("t-wave"))                    return 1;
  return 2; // LVH, LBBB, unknown severe
}

/** Build ML payload — profile has: age, sex, cp, trestbps, chol, restecg, exang, oldpeak
 *  Missing profile columns get safe clinical defaults.
 *  ECG supplies: restecg (from label), thalach (from heart_rate)            */
function buildECGPayload(ecg, profile) {
  return {
    age:         profile.age      ?? 50,    // profile
    sex:         profile.sex      ?? 1,     // profile
    cp:          profile.cp       ?? 0,     // profile
    trestbps:    profile.trestbps ?? 120,   // profile
    chol:        profile.chol     ?? 200,   // profile
    fbs:         0,                         // not in profile → safe default (no high sugar)
    restecg:     labelToRestecg(ecg.label), // ← ECG CNN label
    thalach:     ecg.heart_rate   ?? 150,   // ← ECG measured heart rate
    exang:       profile.exang    ?? 0,     // profile
    oldpeak:     profile.oldpeak  ?? 0,     // profile
    slope:       1,                         // not in profile → normal slope default
    ca:          0,                         // not in profile → no blocked vessels default
    thal:        2,                         // not in profile → normal thal default
    spo2:        97.5,                      // not in profile → normal SpO2 default
    temperature: 37.0,                      // not in profile → normal temp default
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

/** Gemini clinical interpretation */
async function geminiAnalyze(ecg, profile, ml) {
  const flagList    = (ml.flagged_features  || []).map(f => `${f.label}: ${f.value}`).join(", ") || "None";
  const diseaseList = (ml.disease_breakdown || []).map(d => `${d.name} (${d.pct}%)`).join(", ");
  const sexLabel    = (profile.sex === 1 || profile.sex === "1") ? "Male" : "Female";

  const prompt = `
You are a senior cardiologist AI reviewing an ECG-based heart disease risk assessment.

PATIENT:
  Age: ${profile.age}, Sex: ${sexLabel}
  BP: ${profile.trestbps} mmHg, Cholesterol: ${profile.chol} mg/dL
  Chest Pain Type: ${profile.cp}, Exercise Angina: ${profile.exang === 1 ? "Yes" : "No"}
  ST Depression (Oldpeak): ${profile.oldpeak ?? "N/A"}

ECG READING  (recorded ${ecg.created_at}):
  Heart Rate: ${ecg.heart_rate ?? "N/A"} bpm
  Sample Rate: ${ecg.sample_rate ?? "N/A"} Hz
  Duration: ${ecg.duration_sec ?? "N/A"} seconds
  CNN Arrhythmia Classification: ${ecg.label ?? "Unknown"}

ML RISK ASSESSMENT:
  Overall Risk: ${ml.risk_percentage}% (${ml.risk_level})
  Disease Breakdown: ${diseaseList}
  Abnormal Readings: ${flagList}

Return ONLY this JSON (no markdown, no preamble):
{
  "summary": "2-3 sentence plain-language assessment referencing the ECG findings",
  "primaryConcern": "single most critical finding from the ECG or vitals",
  "ecgInterpretation": "brief clinical interpretation of the ECG classification",
  "recommendations": ["3-4 actionable clinical steps"],
  "lifestyle": ["2-3 daily habit changes"],
  "urgency": "Routine | Soon | Urgent | Emergency",
  "disclaimer": "one-line medical disclaimer"
}`.trim();

  const model   = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
  const result  = await model.generateContent(prompt);
  const rawText = result.response.text().trim();

  try {
    return JSON.parse(rawText);
  } catch {
    const clean = rawText.replace(/```json|```/gi, "").trim();
    try { return JSON.parse(clean); } catch { return { summary: rawText }; }
  }
}

/* ─── GET /ecg/latest/:user_id ─────────────────────────────────────────── */
router.get("/latest/:user_id", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("ecg_data")
      .select("*")
      .eq("user_id", req.params.user_id)
      .order("created_at", { ascending: false })
      .maybeSingle();

    if (error) return res.status(404).json({ error: error.message });
    if (!data) return res.status(404).json({ error: "No ECG records found for this user yet. Upload an ECG reading to get started." });
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ─── GET /ecg/history/:user_id ────────────────────────────────────────── */
router.get("/history/:user_id", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("ecg_data")
      .select("id, created_at, heart_rate, label, duration_sec, sample_rate")  // no samples
      .eq("user_id", req.params.user_id)
      .order("created_at", { ascending: false })
      .limit(10);

    if (error) return res.status(500).json({ error: error.message });
    res.json(data || []);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ─── GET /ecg/record/:id ──────────────────────────────────────────────── */
router.get("/record/:id", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("ecg_data")
      .select("*")
      .eq("id", req.params.id)
        .maybeSingle();  

    if (error) return res.status(404).json({ error: error.message });
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


/* ─── POST /ecg/insert ─────────────────────────────────────────────────── */
router.post("/insert", async (req, res) => {
  try {
    const { user_id, samples, sample_rate, heart_rate, label, duration_sec } = req.body;

    if (!user_id || !samples) {
      return res.status(400).json({ error: "user_id and samples are required." });
    }

    const parsedSamples = typeof samples === "string" ? JSON.parse(samples) : samples;

    const { data, error } = await supabase
      .from("ecg_data")
      .insert([{
        user_id,
        samples:      parsedSamples,
        sample_rate:  sample_rate  || 360,
        heart_rate:   heart_rate   || null,
        label:        label        || "Normal",
        duration_sec: duration_sec || (parsedSamples.length / (sample_rate || 360)),
      }])
      .select()
      .single();

    if (error) return res.status(500).json({ error: error.message });
    res.json({ success: true, record: data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ─── POST /ecg/predict/:user_id ───────────────────────────────────────── */
router.post("/predict/:user_id", async (req, res) => {
  const { user_id } = req.params;
  // Optional: pass record_id in body to predict from a specific ECG
  const { record_id } = req.body || {};

  try {
    // 1. Fetch ECG record
    let ecgQuery = supabase.from("ecg_data").select("*").eq("user_id", user_id);
    if (record_id) {
      ecgQuery = ecgQuery.eq("id", record_id);
    } else {
      ecgQuery = ecgQuery.order("created_at", { ascending: false }).limit(1);
    }
    const { data: ecgRows, error: ecgErr } = await ecgQuery;
    if (ecgErr || !ecgRows?.length) {
      return res.status(404).json({ error: "No ECG record found for this user." });
    }
    const ecg = ecgRows[0];

    // 2. Fetch user profile — only columns that exist in the table
    const { data: profileData, error: profileErr } = await supabase
      .from("profile")
      .select("age, sex, cp, trestbps, chol, restecg, exang, oldpeak")
      .eq("id", user_id)
      .maybeSingle();

    if (profileErr) {
      return res.status(500).json({ error: "Profile fetch failed: " + profileErr.message });
    }

    // If no profile exists, fall back to clinical defaults (buildECGPayload handles nulls via ??)
    const profile = profileData || {};
    const profileMissing = !profileData;

    // 3. Build payload — profile fields + ECG-derived fields + safe defaults
    const payload = buildECGPayload(ecg, profile);

    // 4. ML prediction
    const ml = await mlPredict(payload);

    // 5. Gemini analysis
    const gemini = await geminiAnalyze(ecg, profile, ml);

    res.json({
      ecg_record:   { id: ecg.id, created_at: ecg.created_at, heart_rate: ecg.heart_rate, label: ecg.label, duration_sec: ecg.duration_sec, sample_rate: ecg.sample_rate },
      payload_used: payload,
      profile_missing: profileMissing,
      ...ml,
      gemini,
    });
  } catch (err) {
    console.error("[ecg.routes] predict error:", err.message);
    res.status(500).json({ error: "ECG prediction failed", detail: err.message });
  }
});

module.exports = router;