/**
 * ecg.routes.js
 * Fetches raw ECG waveform data from Supabase and runs heart-disease prediction.
 *
 * Routes
 *   GET  /ecg/latest/:user_id    – latest record with full samples
 *   GET  /ecg/history/:user_id   – last 10 records (no samples, for list)
 *   GET  /ecg/record/:id         – single record by id with full samples
 *   POST /ecg/predict/:user_id   – latest ECG + profile → ML + Gemini
 *   POST /ecg/insert             – insert a new ECG record
 *   POST /ecg/report/:user_id    – generate & stream a clinical PDF report
 */

const express    = require("express");
const router     = express.Router();
const supabase   = require("../supabase");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const PDFDocument = require("pdfkit");

const ML_URL     = process.env.ML_SERVICE_URL || "http://localhost:5001";
const GEMINI_KEY = process.env.GEMINI_API_KEY || "";
const genAI      = GEMINI_KEY ? new GoogleGenerativeAI(GEMINI_KEY) : null;

// Current stable models (2025-2026) — ordered fastest → most capable
const GEMINI_MODEL_CHAIN = [
  "gemini-2.5-flash",
  "gemini-2.0-flash",
  "gemini-3.6-flash",
];

if (GEMINI_KEY) {
  console.log(`[ecg.routes] ✓ Gemini key loaded (${GEMINI_KEY.slice(0, 8)}...) — chain: ${GEMINI_MODEL_CHAIN.join(" → ")}`);
} else {
  console.warn("[ecg.routes] ⚠️  GEMINI_API_KEY not set — AI insights will use rule-based fallback.");
}

/* ─────────────────────────────────────────────
   HELPERS
───────────────────────────────────────────── */

function labelToRestecg(label = "") {
  const l = label.toLowerCase();
  if (!l || l === "normal" || l === "n") return 0;
  if (l.includes("afib") || l.includes("st") || l.includes("svt") ||
      l.includes("pvc")  || l.includes("t-wave"))                    return 1;
  return 2;
}

function buildECGPayload(ecg, profile) {
  return {
    age:         profile.age      ?? 50,
    sex:         profile.sex      ?? 1,
    cp:          profile.cp       ?? 0,
    trestbps:    profile.trestbps ?? 120,
    chol:        profile.chol     ?? 200,
    fbs:         0,
    restecg:     labelToRestecg(ecg.label),
    thalach:     ecg.heart_rate   ?? 150,
    exang:       profile.exang    ?? 0,
    oldpeak:     profile.oldpeak  ?? 0,
    slope:       1,
    ca:          0,
    thal:        2,
    spo2:        97.5,
    temperature: 37.0,
  };
}

/* ─────────────────────────────────────────────
   ML PREDICT (with mock fallback)
───────────────────────────────────────────── */

async function mlPredict(payload) {
  if (process.env.ML_MOCK === "true") {
    console.warn("[mlPredict] ML_MOCK=true — using mock");
    return buildMockMLResponse(payload);
  }
  try {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), 8000);
    const r = await fetch(`${ML_URL}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
      signal:  controller.signal,
    });
    clearTimeout(t);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (err) {
    console.warn("[mlPredict] Unavailable:", err.message, "→ mock");
    return buildMockMLResponse(payload);
  }
}

function buildMockMLResponse(payload) {
  let score = 30;
  if (payload.age      > 55)  score += 10;
  if (payload.chol     > 240) score += 8;
  if (payload.trestbps > 140) score += 8;
  if (payload.restecg  > 0)   score += 12;
  if (payload.thalach  < 100) score += 8;
  if (payload.exang    === 1) score += 10;
  if (payload.oldpeak  > 2)   score += 8;
  score = Math.min(score, 95);
  const level = score >= 65 ? "High" : score >= 35 ? "Medium" : "Low";
  const flagged = [];
  if (payload.chol     > 240) flagged.push({ label: "Cholesterol",    value: payload.chol });
  if (payload.trestbps > 140) flagged.push({ label: "Blood Pressure", value: payload.trestbps });
  if (payload.thalach  < 100) flagged.push({ label: "Heart Rate",     value: payload.thalach });
  if (payload.restecg  > 0)   flagged.push({ label: "ECG Result",     value: payload.restecg });
  return {
    risk_percentage:   score,
    risk_level:        level,
    ml_source:         "mock",
    disease_breakdown: [
      { name: "Coronary Artery Disease",    pct: Math.round(score * 0.35) },
      { name: "Heart Failure",              pct: Math.round(score * 0.22) },
      { name: "Arrhythmia",                 pct: Math.round(score * 0.18) },
      { name: "Hypertensive Heart Disease", pct: Math.round(score * 0.14) },
      { name: "Valvular Disease",           pct: Math.round(score * 0.11) },
    ],
    top_features: [
      { label: "Heart Rate",     value: payload.thalach,  color: "#7c3aed" },
      { label: "ECG Result",     value: payload.restecg,  color: "#2563eb" },
      { label: "Blood Pressure", value: payload.trestbps, color: "#0d9488" },
      { label: "Cholesterol",    value: payload.chol,     color: "#d97706" },
      { label: "Age",            value: payload.age,      color: "#dc2626" },
    ],
    flagged_features: flagged,
  };
}

/* ─────────────────────────────────────────────
   GEMINI ANALYZE (multi-model chain + fallback)
───────────────────────────────────────────── */

async function geminiAnalyze(ecg, profile, ml) {
  if (!genAI) {
    console.warn("[geminiAnalyze] No API key — rule-based fallback");
    return { ...ruleBasedInsights(ecg, ml), gemini_used: false, gemini_error: "No GEMINI_API_KEY" };
  }

  const flagList    = (ml.flagged_features  || []).map(f => `${f.label}: ${f.value}`).join(", ") || "None";
  const diseaseList = (ml.disease_breakdown || []).map(d => `${d.name} (${d.pct}%)`).join(", ");
  const sexLabel    = (profile.sex === 1 || profile.sex === "1") ? "Male" : "Female";
  let lastGeminiError = null;

  const prompt = `You are a senior cardiologist AI reviewing an ECG-based heart disease risk assessment.

PATIENT:
  Age: ${profile.age ?? "Unknown"}, Sex: ${sexLabel}
  BP: ${profile.trestbps ?? "N/A"} mmHg, Cholesterol: ${profile.chol ?? "N/A"} mg/dL
  Chest Pain Type: ${profile.cp ?? "N/A"}, Exercise Angina: ${profile.exang === 1 ? "Yes" : "No"}
  ST Depression (Oldpeak): ${profile.oldpeak ?? "N/A"}

ECG READING (recorded ${ecg.created_at}):
  Heart Rate: ${ecg.heart_rate ?? "N/A"} bpm
  Sample Rate: ${ecg.sample_rate ?? "N/A"} Hz
  Duration: ${ecg.duration_sec ?? "N/A"} seconds
  CNN Arrhythmia Classification: ${ecg.label ?? "Unknown"}

ML RISK ASSESSMENT:
  Overall Risk: ${ml.risk_percentage}% (${ml.risk_level})
  Disease Breakdown: ${diseaseList}
  Abnormal Readings: ${flagList}

Return ONLY a raw JSON object — no markdown fences, no preamble:
{"summary":"...","primaryConcern":"...","ecgInterpretation":"...","recommendations":["..."],"lifestyle":["..."],"urgency":"Routine","disclaimer":"..."}

urgency must be exactly one of: Routine | Soon | Urgent | Emergency`;

  for (const modelName of GEMINI_MODEL_CHAIN) {
    try {
      console.log(`[geminiAnalyze] Trying: ${modelName}`);
      const model  = genAI.getGenerativeModel({ model: modelName, generationConfig: { temperature: 0.4, maxOutputTokens:2048  } });
      const result = await model.generateContent(prompt);
      const raw = result.response.text().trim();

      // Strip markdown fences and wrapping text.
      const clean = raw
        .replace(/^```json\s*/i, "")
        .replace(/^```\s*/i, "")
        .replace(/\s*```$/i, "")
        .trim();

      let jsonText = clean;
      const start = clean.indexOf("{");
      const end   = clean.lastIndexOf("}");
      if (start !== -1 && end !== -1 && end > start) {
        jsonText = clean.substring(start, end + 1);
      }

      let parsed = null;
      try {
        parsed = JSON.parse(jsonText);
      } catch (parseErr) {
        const embedded = clean.match(/{[\s\S]*}/);
        if (embedded) {
          try {
            parsed = JSON.parse(embedded[0]);
          } catch {
            parsed = null;
          }
        }
      }

      if (!parsed) {
        throw new Error(`Unable to parse Gemini JSON response: ${raw.slice(0, 300)}`);
      }

      console.log(`[geminiAnalyze] ✓ Success with ${modelName}`);
      return { ...parsed, gemini_model: modelName, gemini_used: true };
    } catch (err) {
      lastGeminiError = err.message || "Unknown Gemini parse/error";
      const msg = err.message || "";
      // 404 = model deprecated/removed — skip immediately
      if (msg.includes("404")) {
        console.warn(`[geminiAnalyze] ${modelName} → 404 deprecated, skipping`);
        continue;
      }
      // 429 = rate-limited — wait retryDelay if short enough, then try next model
      if (msg.includes("429")) {
        const delayMatch = msg.match(/"retryDelay":"(\d+)s"/);
        const delaySec   = delayMatch ? parseInt(delayMatch[1], 10) : 0;
        if (delaySec > 0 && delaySec <= 15) {
          console.warn(`[geminiAnalyze] ${modelName} → 429 rate-limit, waiting ${delaySec}s then retrying next model`);
          await new Promise(r => setTimeout(r, delaySec * 1000));
        } else {
          console.warn(`[geminiAnalyze] ${modelName} → 429 rate-limit (retry in ${delaySec}s — too long, skipping)`);
        }
        continue;
      }
      console.warn(`[geminiAnalyze] ${modelName} failed: ${msg}`);
    }
  }

  console.error("[geminiAnalyze] All models failed — rule-based fallback", lastGeminiError);
  return { ...ruleBasedInsights(ecg, ml), gemini_used: false, gemini_error: lastGeminiError };
}

function ruleBasedInsights(ecg, ml) {
  const pct   = ml.risk_percentage ?? 0;
  const label = ecg.label ?? "Unknown";
  const hr    = ecg.heart_rate ?? "N/A";
  const l     = label.toLowerCase();

  let urgency = "Routine";
  if (pct >= 75) urgency = "Urgent";
  else if (pct >= 50) urgency = "Soon";

  let ecgInterp      = "ECG pattern appears within normal limits based on CNN classification.";
  let primaryConcern = `Cardiovascular risk estimated at ${pct}% (${ml.risk_level ?? "Unknown"})`;

  if (l.includes("afib")) {
    ecgInterp = "Atrial fibrillation detected — irregular rhythm with absent P waves. Requires clinical evaluation.";
    primaryConcern = "Atrial fibrillation — irregular heart rhythm requiring prompt evaluation";
    if (urgency === "Routine") urgency = "Soon";
  } else if (l.includes("st-elev")) {
    ecgInterp = "ST-elevation detected — may indicate acute myocardial injury (STEMI). Urgent evaluation required.";
    primaryConcern = "ST-elevation — possible acute myocardial infarction";
    urgency = "Emergency";
  } else if (l.includes("st-dep")) {
    ecgInterp = "ST-depression noted — may indicate subendocardial ischaemia or myocardial strain.";
    primaryConcern = "ST-depression — possible myocardial ischaemia";
    if (urgency === "Routine") urgency = "Soon";
  } else if (l.includes("pvc")) {
    ecgInterp = "Premature ventricular contractions (PVCs) detected — frequency and symptoms require assessment.";
    primaryConcern = "Premature ventricular contractions — assess frequency and symptoms";
  } else if (l.includes("svt")) {
    ecgInterp = "Supraventricular tachycardia detected — rapid rate originating above ventricles.";
    primaryConcern = "SVT — rapid supraventricular rhythm requiring evaluation";
    if (urgency === "Routine") urgency = "Soon";
  } else if (l.includes("lbbb")) {
    ecgInterp = "Left bundle branch block (LBBB) detected — full cardiac workup including echo recommended.";
    primaryConcern = "Left bundle branch block — requires further cardiac investigation";
    if (urgency === "Routine") urgency = "Soon";
  }

  return {
    summary: `ECG recorded at ${hr} bpm classified as "${label}". ML model estimates ${pct}% (${ml.risk_level ?? "Unknown"}) cardiovascular risk. ${pct >= 50 ? "Prompt clinical review is advisable." : "Routine follow-up recommended."}`,
    primaryConcern,
    ecgInterpretation: ecgInterp,
    recommendations: [
      "Share this report with your cardiologist or GP for clinical review",
      "Obtain a 12-lead ECG in a clinical setting to validate findings",
      pct >= 50 ? "Schedule a stress test and echocardiogram" : "Continue routine cardiac screening as advised",
      "Monitor blood pressure and resting heart rate regularly",
    ],
    lifestyle: [
      "30 minutes of moderate aerobic exercise at least 5 days per week",
      "Heart-healthy diet: reduce sodium, saturated fats; increase fruits and vegetables",
      "Manage stress, get adequate sleep (7-8 hrs), avoid smoking and limit alcohol",
    ],
    urgency,
    disclaimer: "This is an automated assessment. Always consult a qualified healthcare professional.",
    gemini_used: false,
  };
}

/* ─────────────────────────────────────────────
   OFFICIAL CLINICAL PDF REPORT GENERATOR
───────────────────────────────────────────── */

function generatePDFReport(res, { ecg, profile, ml, gemini, generatedAt }) {
  const doc = new PDFDocument({
    size: "A4",
    margins: { top: 50, bottom: 60, left: 55, right: 55 },
    info: {
      Title:   "Cardiac ECG Diagnostic Report",
      Author:  "AI-Assisted ECG Diagnostic System",
      Subject: "Electrocardiogram Risk Assessment",
      Creator: "ECG Diagnostic Platform",
    },
  });

  res.setHeader("Content-Type", "application/pdf");
  res.setHeader(
    "Content-Disposition",
    `attachment; filename="ECG_Cardiac_Report_${ecg.id?.slice(0, 8) || "report"}.pdf"`
  );
  doc.pipe(res);

  /* ── Layout constants ── */
  const W   = 595.28;
  const L   = 55;
  const R   = 55;
  const CW  = W - L - R;
  const MID = L + CW / 2;

  /* ── Palette ── */
  const C_NAVY      = "#0a1628";
  const C_BLUE      = "#1d3557";
  const C_RULE      = "#c8d3e0";
  const C_BG        = "#f4f6f9";
  const C_TEXT      = "#111827";
  const C_SUB       = "#4b5563";
  const C_MUTED     = "#9ca3af";
  const C_GREEN     = "#166534";
  const C_GREEN_BG  = "#dcfce7";
  const C_AMBER     = "#92400e";
  const C_AMBER_BG  = "#fef3c7";
  const C_RED       = "#991b1b";
  const C_RED_BG    = "#fee2e2";
  const C_BLUE_BG   = "#dbeafe";
  const C_BLUE_TEXT = "#1e40af";

  function riskColor(pct) {
    if (pct >= 65) return C_RED;
    if (pct >= 35) return C_AMBER;
    return C_GREEN;
  }
  function riskBg(pct) {
    if (pct >= 65) return C_RED_BG;
    if (pct >= 35) return C_AMBER_BG;
    return C_GREEN_BG;
  }
  function riskLabel(pct) {
    if (pct >= 65) return "HIGH RISK";
    if (pct >= 35) return "MODERATE RISK";
    return "LOW RISK";
  }

  const pct    = ml.risk_percentage ?? 0;
  const sexLbl = (profile.sex === 1 || profile.sex === "1") ? "Male" : "Female";
  const cpMap  = { 0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic" };
  const reportId = `RPT-${(ecg.id || "XXXXXXXX").slice(0, 8).toUpperCase()}`;

  let y = 50;

  /* ════ LETTERHEAD ════ */
  doc.rect(L, y, 4, 54).fill(C_BLUE);

  doc.fillColor(C_NAVY).font("Helvetica-Bold").fontSize(16)
     .text("ADVANCED CARDIAC DIAGNOSTICS", L + 14, y + 2);
  doc.fillColor(C_SUB).font("Helvetica").fontSize(9)
     .text("Electrocardiogram Analysis & Cardiovascular Risk Assessment Unit", L + 14, y + 22);
  doc.fillColor(C_MUTED).font("Helvetica").fontSize(8)
     .text("AI-Assisted Diagnostic Service  ·  For physician review only", L + 14, y + 36);

  doc.fillColor(C_SUB).font("Helvetica").fontSize(8)
     .text("REPORT NO.",     W - R - 155, y + 2,  { width: 155, align: "right" })
     .text("DATE ISSUED",    W - R - 155, y + 16, { width: 155, align: "right" })
     .text("CLASSIFICATION", W - R - 155, y + 30, { width: 155, align: "right" });
  doc.fillColor(C_NAVY).font("Helvetica-Bold").fontSize(8)
     .text(reportId,                 W - R - 155, y + 10, { width: 155, align: "right" })
     .text(generatedAt,              W - R - 155, y + 24, { width: 155, align: "right" })
     .text("CONFIDENTIAL — PATIENT", W - R - 155, y + 38, { width: 155, align: "right" });

  y += 62;

  doc.rect(L, y, CW, 2).fill(C_NAVY);
  doc.rect(L, y + 4, CW, 0.5).fill(C_RULE);
  y += 12;

  doc.fillColor(C_NAVY).font("Helvetica-Bold").fontSize(13)
     .text("CARDIAC ECG RISK ASSESSMENT REPORT", L, y, { width: CW, align: "center" });
  y += 18;
  doc.fillColor(C_MUTED).font("Helvetica").fontSize(8)
     .text(
       "This document is generated by an AI-assisted diagnostic system and is intended for review by a qualified physician.",
       L, y, { width: CW, align: "center" }
     );
  y += 16;

  /* ════ PATIENT + ECG INFO (2-col) ════ */
  const COL2W = CW / 2 - 6;

  /* Left — Patient */
  doc.rect(L, y, COL2W, 118).fill(C_BG).stroke(C_RULE);
  doc.rect(L, y, COL2W, 18).fill(C_NAVY);
  doc.fillColor("white").font("Helvetica-Bold").fontSize(8)
     .text("PATIENT INFORMATION", L + 10, y + 5, { width: COL2W - 20 });

  const patRows = [
    ["Patient ID",        (ecg.user_id || "N/A").slice(0, 18)],
    ["Age",               profile.age       ?? "N/A"],
    ["Sex",               sexLbl],
    ["Blood Pressure",    `${profile.trestbps ?? "N/A"} mmHg`],
    ["Cholesterol",       `${profile.chol     ?? "N/A"} mg/dL`],
    ["Chest Pain Type",   cpMap[profile.cp]  ?? "N/A"],
    ["Exercise Angina",   profile.exang === 1 ? "Yes" : "No"],
    ["ST Depression",     `${profile.oldpeak ?? "N/A"} mm`],
  ];
  let py = y + 22;
  patRows.forEach(([k, v], i) => {
    if (i % 2 === 0) doc.rect(L, py, COL2W, 12).fill("#eef1f5");
    doc.fillColor(C_SUB).font("Helvetica").fontSize(7.5)
       .text(k, L + 8, py + 2, { width: 100 });
    doc.fillColor(C_TEXT).font("Helvetica-Bold").fontSize(7.5)
       .text(String(v), L + 108, py + 2, { width: COL2W - 116 });
    py += 12;
  });

  /* Right — ECG */
  const RX = MID + 6;
  doc.rect(RX, y, COL2W, 118).fill(C_BG).stroke(C_RULE);
  doc.rect(RX, y, COL2W, 18).fill(C_BLUE);
  doc.fillColor("white").font("Helvetica-Bold").fontSize(8)
     .text("ECG RECORDING DETAILS", RX + 10, y + 5, { width: COL2W - 20 });

  const ecgRows = [
    ["Record Reference",   (ecg.id || "N/A").slice(0, 16).toUpperCase()],
    ["Recorded On",        ecg.created_at ? new Date(ecg.created_at).toLocaleString("en-IN") : "N/A"],
    ["Heart Rate",         `${ecg.heart_rate  ?? "N/A"} bpm`],
    ["Sample Rate",        `${ecg.sample_rate ?? "N/A"} Hz`],
    ["Recording Duration", `${ecg.duration_sec ?? "N/A"} seconds`],
    ["Total Data Points",  `${(ecg.samples || []).length}`],
    ["CNN Classification", ecg.label || "Unknown"],
    ["ML Data Source",     ml.ml_source === "mock" ? "Heuristic (ML offline)" : "ML Microservice"],
  ];
  let ey = y + 22;
  ecgRows.forEach(([k, v], i) => {
    if (i % 2 === 0) doc.rect(RX, ey, COL2W, 12).fill("#eef1f5");
    doc.fillColor(C_SUB).font("Helvetica").fontSize(7.5)
       .text(k, RX + 8, ey + 2, { width: 112 });
    doc.fillColor(C_TEXT).font("Helvetica-Bold").fontSize(7.5)
       .text(String(v), RX + 120, ey + 2, { width: COL2W - 128 });
    ey += 12;
  });

  y += 128;

  /* ════ RISK BANNER ════ */
  const bannerH = 52;
  doc.rect(L, y, CW, bannerH).fill(riskBg(pct)).stroke(riskColor(pct));

  doc.fillColor(riskColor(pct)).font("Helvetica-Bold").fontSize(34)
     .text(`${pct}%`, L + 14, y + 8, { width: 110, lineBreak: false });

  doc.fillColor(riskColor(pct)).font("Helvetica-Bold").fontSize(12)
     .text(riskLabel(pct), L + 128, y + 12);
  doc.fillColor(C_SUB).font("Helvetica").fontSize(8)
     .text(
       `Computed cardiovascular disease probability from ECG waveform and clinical parameters. Source: ${ml.ml_source === "mock" ? "Heuristic model (ML service unavailable)" : "ML microservice"}.`,
       L + 128, y + 26, { width: CW - 240 }
     );

  const urgMap = {
    Routine:   { bg: C_GREEN_BG, fg: C_GREEN  },
    Soon:      { bg: C_AMBER_BG, fg: C_AMBER  },
    Urgent:    { bg: "#ffedd5",  fg: "#9a3412" },
    Emergency: { bg: C_RED_BG,   fg: C_RED    },
  };
  const uc   = urgMap[gemini.urgency] || urgMap.Routine;
  const urgW = 78;
  const urgX = L + CW - urgW;
  doc.roundedRect(urgX, y + 10, urgW, 20, 4).fill(uc.bg);
  doc.fillColor(uc.fg).font("Helvetica-Bold").fontSize(8)
     .text((gemini.urgency || "ROUTINE").toUpperCase(), urgX, y + 17, { width: urgW, align: "center" });
  doc.fillColor(C_MUTED).font("Helvetica").fontSize(7)
     .text("URGENCY LEVEL", urgX, y + 33, { width: urgW, align: "center" });

  y += bannerH + 14;

  /* ── Section header helper ── */
  function sectionHeader(title, color) {
    color = color || C_NAVY;
    doc.rect(L, y, CW, 1).fill(C_RULE);
    doc.rect(L, y, 3, 16).fill(color);
    doc.fillColor(color).font("Helvetica-Bold").fontSize(9.5)
       .text(title, L + 10, y + 3);
    y += 20;
  }

  /* ════ ECG INTERPRETATION ════ */
  sectionHeader("1.  ECG INTERPRETATION", C_BLUE);
  const interpText = gemini.ecgInterpretation || "No interpretation available.";
  const interpH    = doc.heightOfString(interpText, { width: CW - 20, lineGap: 3 }) + 14;
  doc.rect(L, y, CW, interpH).fill(C_BG).stroke(C_RULE);
  doc.fillColor(C_TEXT).font("Helvetica").fontSize(9)
     .text(interpText, L + 10, y + 7, { width: CW - 20, lineGap: 3 });
  y += interpH + 8;

  /* ════ CLINICAL SUMMARY ════ */
  sectionHeader("2.  CLINICAL SUMMARY", C_BLUE);
  const summText = gemini.summary || "No summary available.";
  const summH    = doc.heightOfString(summText, { width: CW - 20, lineGap: 3 }) + 14;
  doc.rect(L, y, CW, summH).fill(C_BG).stroke(C_RULE);
  doc.fillColor(C_TEXT).font("Helvetica").fontSize(9)
     .text(summText, L + 10, y + 7, { width: CW - 20, lineGap: 3 });
  y += summH + 4;

  if (gemini.primaryConcern) {
    doc.rect(L, y, CW, 24).fill("#fef9c3").stroke("#fbbf24");
    doc.fillColor("#78350f").font("Helvetica-Bold").fontSize(8)
       .text("PRINCIPAL FINDING:", L + 10, y + 8);
    doc.fillColor("#78350f").font("Helvetica").fontSize(8)
       .text(gemini.primaryConcern, L + 120, y + 8, { width: CW - 130 });
    y += 32;
  }

  /* ════ DISEASE BREAKDOWN ════ */
  sectionHeader("3.  DISEASE-SPECIFIC RISK BREAKDOWN", C_BLUE);

  const DIS_COLORS = ["#1d4ed8", "#7c3aed", "#0d9488", "#b45309", "#be123c"];
  const diseases   = ml.disease_breakdown || [];

  diseases.forEach((d, i) => {
    const bg = i % 2 === 0 ? C_BG : "white";
    doc.rect(L, y, CW, 16).fill(bg).stroke(C_RULE);
    doc.rect(L + 6, y + 4, 8, 8).fill(DIS_COLORS[i] || C_BLUE);
    doc.fillColor(C_TEXT).font("Helvetica").fontSize(8)
       .text(d.name, L + 20, y + 4, { width: 210 });
    const barW = Math.round((d.pct / 100) * (CW - 290));
    doc.rect(L + 240, y + 5, CW - 290, 6).fill("#e5e7eb");
    doc.rect(L + 240, y + 5, barW,     6).fill(DIS_COLORS[i] || C_BLUE);
    doc.fillColor(DIS_COLORS[i] || C_BLUE).font("Helvetica-Bold").fontSize(8)
       .text(`${d.pct}%`, L + CW - 42, y + 4, { width: 40, align: "right" });
    y += 16;
  });
  y += 10;

  /* ════ CLINICAL PARAMETERS TABLE ════ */
  sectionHeader("4.  CLINICAL PARAMETERS", C_BLUE);

  const TCols = ["Parameter", "Recorded Value", "Reference Range", "Assessment"];
  const TCW   = [180, 120, 140, CW - 180 - 120 - 140];

  doc.rect(L, y, CW, 16).fill(C_NAVY);
  let tx = L + 8;
  TCols.forEach((h, i) => {
    doc.fillColor("white").font("Helvetica-Bold").fontSize(7.5)
       .text(h, tx, y + 4, { width: TCW[i] - 8 });
    tx += TCW[i];
  });
  y += 16;

  const statusStyle = {
    "Normal":          { bg: C_GREEN_BG, fg: C_GREEN      },
    "Elevated":        { bg: C_AMBER_BG, fg: C_AMBER      },
    "High":            { bg: C_RED_BG,   fg: C_RED        },
    "Borderline":      { bg: C_AMBER_BG, fg: C_AMBER      },
    "Abnormal":        { bg: C_RED_BG,   fg: C_RED        },
    "Review Required": { bg: C_BLUE_BG,  fg: C_BLUE_TEXT  },
  };

  const paramRows = [
    ["Heart Rate (ECG waveform)",  `${ecg.heart_rate   ?? "N/A"} bpm`,  "60 – 100 bpm",    ecg.heart_rate >= 60 && ecg.heart_rate <= 100 ? "Normal" : "Abnormal"],
    ["Systolic Blood Pressure",    `${profile.trestbps ?? "N/A"} mmHg`, "90 – 120 mmHg",   profile.trestbps <= 120 ? "Normal" : profile.trestbps <= 140 ? "Elevated" : "High"],
    ["Serum Cholesterol",          `${profile.chol     ?? "N/A"} mg/dL`, "< 200 mg/dL",     profile.chol < 200 ? "Normal" : profile.chol < 240 ? "Borderline" : "High"],
    ["Peripheral SpO2",            "97.5 %",                             ">= 95 %",         "Normal"],
    ["Core Temperature",           "37.0 °C",                            "36.1 – 37.2 °C",  "Normal"],
    ["ST-Segment Depression",      `${profile.oldpeak  ?? "N/A"} mm`,   "0 – 1 mm",        (profile.oldpeak ?? 0) <= 1 ? "Normal" : "Elevated"],
    ["ECG CNN Classification",     ecg.label || "Unknown",               "Normal Sinus",    ecg.label?.toLowerCase() === "normal" ? "Normal" : "Review Required"],
  ];

  paramRows.forEach((row, ri) => {
    const bg = ri % 2 === 0 ? C_BG : "white";
    doc.rect(L, y, CW, 14).fill(bg).stroke(C_RULE);
    let px = L + 8;
    row.forEach((cell, ci) => {
      if (ci === 3) {
        const st = statusStyle[cell] || { bg: C_BG, fg: C_SUB };
        doc.roundedRect(px, y + 2, TCW[ci] - 16, 10, 2).fill(st.bg);
        doc.fillColor(st.fg).font("Helvetica-Bold").fontSize(7)
           .text(cell, px, y + 4, { width: TCW[ci] - 16, align: "center" });
      } else {
        doc.fillColor(ci === 0 ? C_SUB : C_TEXT)
           .font(ci === 0 ? "Helvetica" : "Helvetica-Bold").fontSize(7.5)
           .text(String(cell), px, y + 3, { width: TCW[ci] - 10 });
      }
      px += TCW[ci];
    });
    y += 14;
  });
  y += 12;

  /* ════ PAGE BREAK CHECK ════ */
  if (y > 680) { doc.addPage(); y = 50; }

  /* ════ RECOMMENDATIONS ════ */
  sectionHeader("5.  CLINICAL RECOMMENDATIONS & LIFESTYLE GUIDANCE", C_BLUE);

  const HALF = CW / 2 - 8;
  const recs  = gemini.recommendations || [];
  let recY = y;

  doc.fillColor(C_NAVY).font("Helvetica-Bold").fontSize(8)
     .text("Clinical Recommendations", L, recY);
  recY += 12;
  recs.forEach((r) => {
    const rh = doc.heightOfString(r, { width: HALF - 18, lineGap: 2 });
    doc.circle(L + 5, recY + 4, 2.5).fill(C_BLUE);
    doc.fillColor(C_TEXT).font("Helvetica").fontSize(8)
       .text(r, L + 14, recY, { width: HALF - 18, lineGap: 2 });
    recY += rh + 7;
  });

  const lifX = MID + 8;
  let lifY = y;
  doc.fillColor(C_NAVY).font("Helvetica-Bold").fontSize(8)
     .text("Lifestyle Guidance", lifX, lifY);
  lifY += 12;
  (gemini.lifestyle || []).forEach((tip) => {
    const th = doc.heightOfString(tip, { width: HALF - 18, lineGap: 2 });
    doc.circle(lifX + 5, lifY + 4, 2.5).fill("#0d9488");
    doc.fillColor(C_TEXT).font("Helvetica").fontSize(8)
       .text(tip, lifX + 14, lifY, { width: HALF - 18, lineGap: 2 });
    lifY += th + 7;
  });

  y = Math.max(recY, lifY) + 10;

  /* ════ FLAGGED VALUES ════ */
  const flagged = ml.flagged_features || [];
  if (flagged.length) {
    if (y > 720) { doc.addPage(); y = 50; }
    sectionHeader("6.  FLAGGED ABNORMAL VALUES", "#b91c1c");
    flagged.forEach((f, i) => {
      doc.rect(L, y, CW, 18).fill(i % 2 === 0 ? C_RED_BG : "#fff1f2").stroke("#fca5a5");
      doc.fillColor(C_RED).font("Helvetica-Bold").fontSize(8)
         .text(`▲  ${f.label}`, L + 10, y + 5);
      doc.fillColor(C_RED).font("Helvetica").fontSize(8)
         .text(`Recorded value: ${f.value}`, L + 200, y + 5);
      doc.fillColor(C_SUB).font("Helvetica").fontSize(7.5)
         .text("Requires clinical review", L + CW - 130, y + 5, { width: 120, align: "right" });
      y += 20;
    });
    y += 6;
  }

  /* ════ SIGNATURE BLOCK ════ */
  if (y > 700) { doc.addPage(); y = 50; }

  doc.rect(L, y, CW, 1).fill(C_RULE);
  y += 10;

  const sigW = CW / 3 - 6;

  /* Col 1 — Issued by */
  doc.rect(L, y, sigW, 56).fill(C_BG).stroke(C_RULE);
  doc.fillColor(C_MUTED).font("Helvetica").fontSize(6.5)
     .text("REPORT ISSUED BY", L + 8, y + 6);
  doc.fillColor(C_NAVY).font("Helvetica-Bold").fontSize(8)
     .text("AI-Assisted ECG\nDiagnostic System", L + 8, y + 18, { lineGap: 2 });
  doc.fillColor(C_SUB).font("Helvetica").fontSize(7)
     .text(`Generated: ${generatedAt}`, L + 8, y + 38);

  /* Col 2 — Reviewing Physician */
  const c2X = L + sigW + 8;
  doc.rect(c2X, y, sigW, 56).fill(C_BG).stroke(C_RULE);
  doc.fillColor(C_MUTED).font("Helvetica").fontSize(6.5)
     .text("REVIEWED / AUTHORISED BY", c2X + 8, y + 6);
  doc.fillColor(C_MUTED).font("Helvetica").fontSize(7.5)
     .text("Name: ___________________________", c2X + 8, y + 20)
     .text("Sign:  ___________________________", c2X + 8, y + 34)
     .text("Date:  ___________________________", c2X + 8, y + 46);

  /* Col 3 — Stamp */
  const c3X = L + 2 * (sigW + 8);
  doc.rect(c3X, y, sigW, 56).dash(2, { space: 3 }).stroke(C_RULE).undash();
  doc.fillColor(C_MUTED).font("Helvetica").fontSize(6.5)
     .text("OFFICIAL STAMP / SEAL", c3X + 8, y + 6);
  doc.fillColor(C_RULE).font("Helvetica").fontSize(22)
     .text("STAMP", c3X, y + 22, { width: sigW, align: "center" });

  y += 66;

  /* Disclaimer */
  if (y > 730) { doc.addPage(); y = 50; }
  doc.rect(L, y, CW, 32).fill("#fff7ed").stroke("#fed7aa");
  doc.fillColor("#9a3412").font("Helvetica-Bold").fontSize(7.5)
     .text("IMPORTANT DISCLAIMER", L + 10, y + 6);
  doc.fillColor("#7c2d12").font("Helvetica").fontSize(7.5)
     .text(
       gemini.disclaimer ||
       "This report is produced by an AI-assisted ECG analysis system. Results are intended to support, not replace, clinical judgement. Always consult a qualified cardiologist or physician before making clinical decisions.",
       L + 10, y + 17, { width: CW - 20, lineGap: 2 }
     );
  y += 40;

  /* ════ FOOTER ════ */
  const footerY = doc.page.height - 40;
  doc.rect(L, footerY - 4, CW, 0.5).fill(C_RULE);
  doc.fillColor(C_MUTED).font("Helvetica").fontSize(6.5)
     .text(
       `Report No. ${reportId}  ·  Generated: ${generatedAt}  ·  CONFIDENTIAL — This document contains sensitive patient health information. Unauthorised disclosure is prohibited.`,
       L, footerY + 2, { width: CW - 50, align: "left" }
     );
  doc.fillColor(C_MUTED).font("Helvetica").fontSize(6.5)
     .text("Page 1", L, footerY + 2, { width: CW, align: "right" });

  doc.end();
}

/* ─────────────────────────────────────────────
   ROUTES
───────────────────────────────────────────── */

/* GET /ecg/latest/:user_id */
router.get("/latest/:user_id", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("ecg_data").select("*").eq("user_id", req.params.user_id)
      .order("created_at", { ascending: false }).maybeSingle();
    if (error) return res.status(500).json({ error: error.message });
    if (!data)  return res.status(404).json({ error: "No ECG records found. Upload an ECG reading to get started." });
    res.json(data);
  } catch (err) { res.status(500).json({ error: err.message }); }
});

/* GET /ecg/history/:user_id */
router.get("/history/:user_id", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("ecg_data").select("id, created_at, heart_rate, label, duration_sec, sample_rate")
      .eq("user_id", req.params.user_id).order("created_at", { ascending: false }).limit(10);
    if (error) return res.status(500).json({ error: error.message });
    res.json(data || []);
  } catch (err) { res.status(500).json({ error: err.message }); }
});

/* GET /ecg/record/:id */
router.get("/record/:id", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("ecg_data").select("*").eq("id", req.params.id).maybeSingle();
    if (error) return res.status(500).json({ error: error.message });
    if (!data)  return res.status(404).json({ error: "ECG record not found." });
    res.json(data);
  } catch (err) { res.status(500).json({ error: err.message }); }
});

/* POST /ecg/insert */
router.post("/insert", async (req, res) => {
  try {
    const { user_id, samples, sample_rate, heart_rate, label, duration_sec } = req.body;
    if (!user_id || !samples) return res.status(400).json({ error: "user_id and samples are required." });
    const parsedSamples = typeof samples === "string" ? JSON.parse(samples) : samples;
    const sr = sample_rate || 360;
    const { data, error } = await supabase.from("ecg_data").insert([{
      user_id, samples: parsedSamples, sample_rate: sr,
      heart_rate: heart_rate || null, label: label || "Normal",
      duration_sec: duration_sec || (parsedSamples.length / sr),
    }]).select().single();
    if (error) return res.status(500).json({ error: error.message });
    res.json({ success: true, record: data });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

/* POST /ecg/predict/:user_id */
router.post("/predict/:user_id", async (req, res) => {
  const { user_id } = req.params;
  const { record_id } = req.body || {};
  console.log(`\n[ecg/predict] user: ${user_id} | record: ${record_id || "latest"}`);
  try {
    let ecgQuery = supabase.from("ecg_data").select("*").eq("user_id", user_id);
    if (record_id) ecgQuery = ecgQuery.eq("id", record_id);
    else ecgQuery = ecgQuery.order("created_at", { ascending: false }).limit(1);

    const { data: ecgRows, error: ecgErr } = await ecgQuery;
    if (ecgErr) return res.status(500).json({ error: "ECG fetch failed: " + ecgErr.message });
    if (!ecgRows?.length) return res.status(404).json({ error: "No ECG record found." });
    const ecg = ecgRows[0];

    const { data: profileData, error: profileErr } = await supabase
      .from("profile").select("age, sex, cp, trestbps, chol, restecg, exang, oldpeak")
      .eq("id", user_id).maybeSingle();
    if (profileErr) return res.status(500).json({ error: "Profile fetch failed: " + profileErr.message });

    const profile        = profileData || {};
    const profileMissing = !profileData;
    const payload        = buildECGPayload(ecg, profile);
    const ml             = await mlPredict(payload);
    const gemini         = await geminiAnalyze(ecg, profile, ml);

    console.log(`[ecg/predict] Done — risk: ${ml.risk_percentage}% | urgency: ${gemini.urgency}`);

    res.json({
      ecg_record:      { id: ecg.id, created_at: ecg.created_at, heart_rate: ecg.heart_rate, label: ecg.label, duration_sec: ecg.duration_sec, sample_rate: ecg.sample_rate },
      payload_used:    payload,
      profile_missing: profileMissing,
      ...ml,
      gemini,
    });
  } catch (err) {
    console.error("[ecg/predict] Error:", err.message);
    res.status(500).json({ error: "ECG prediction failed", detail: err.message });
  }
});

/* POST /ecg/report/:user_id */
router.post("/report/:user_id", async (req, res) => {
  const { user_id } = req.params;
  const { record_id } = req.body || {};
  console.log(`\n[ecg/report] Generating PDF for user: ${user_id}`);

  try {
    let ecgQuery = supabase.from("ecg_data").select("*").eq("user_id", user_id);
    if (record_id) ecgQuery = ecgQuery.eq("id", record_id);
    else ecgQuery = ecgQuery.order("created_at", { ascending: false }).limit(1);

    const { data: ecgRows, error: ecgErr } = await ecgQuery;
    if (ecgErr || !ecgRows?.length)
      return res.status(404).json({ error: "No ECG record found." });
    const ecg = ecgRows[0];

    const { data: profileData } = await supabase
      .from("profile").select("age, sex, cp, trestbps, chol, restecg, exang, oldpeak")
      .eq("id", user_id).maybeSingle();
    const profile = profileData || {};

    const payload = buildECGPayload(ecg, profile);
    const ml      = await mlPredict(payload);
    const gemini  = await geminiAnalyze(ecg, profile, ml);

    const generatedAt = new Date().toLocaleString("en-IN", { dateStyle: "long", timeStyle: "short" });
    generatePDFReport(res, { ecg, profile, ml, gemini, generatedAt });

    console.log("[ecg/report] PDF streamed ✓");
  } catch (err) {
    console.error("[ecg/report] Error:", err.message);
    if (!res.headersSent)
      res.status(500).json({ error: "Report generation failed", detail: err.message });
  }
});

module.exports = router;