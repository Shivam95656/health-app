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

// PDF generation — uses pdfkit (npm install pdfkit)
const PDFDocument = require("pdfkit");

const ML_URL     = process.env.ML_SERVICE_URL || "http://localhost:5001";
const GEMINI_KEY = process.env.GEMINI_API_KEY || "";
const genAI      = GEMINI_KEY ? new GoogleGenerativeAI(GEMINI_KEY) : null;

// Try models in order — if one is overloaded, move to next
const GEMINI_MODEL_CHAIN = [
  "gemini-1.5-flash",
  "gemini-1.5-flash-8b",
  "gemini-1.5-pro",
  "gemini-2.5-flash",
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
    return ruleBasedInsights(ecg, ml);
  }

  const flagList    = (ml.flagged_features  || []).map(f => `${f.label}: ${f.value}`).join(", ") || "None";
  const diseaseList = (ml.disease_breakdown || []).map(d => `${d.name} (${d.pct}%)`).join(", ");
  const sexLabel    = (profile.sex === 1 || profile.sex === "1") ? "Male" : "Female";

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
      const model  = genAI.getGenerativeModel({ model: modelName, generationConfig: { temperature: 0.4, maxOutputTokens: 1024 } });
      const result = await model.generateContent(prompt);
      const raw    = result.response.text().trim();
      const clean  = raw.replace(/^```json\s*/i, "").replace(/^```\s*/i, "").replace(/\s*```$/i, "").trim();
      const parsed = JSON.parse(clean);
      console.log(`[geminiAnalyze] ✓ Success with ${modelName}`);
      return { ...parsed, gemini_model: modelName };
    } catch (err) {
      console.warn(`[geminiAnalyze] ${modelName} failed: ${err.message}`);
    }
  }

  console.error("[geminiAnalyze] All models failed — rule-based fallback");
  return ruleBasedInsights(ecg, ml);
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
   PDF REPORT GENERATOR
───────────────────────────────────────────── */

function generatePDFReport(res, { ecg, profile, ml, gemini, generatedAt }) {
  const doc = new PDFDocument({ size: "A4", margins: { top: 50, bottom: 50, left: 50, right: 50 } });

  // Stream directly to HTTP response
  res.setHeader("Content-Type", "application/pdf");
  res.setHeader("Content-Disposition", `attachment; filename="HeartGuard_ECG_Report_${ecg.id?.slice(0,8) || "report"}.pdf"`);
  doc.pipe(res);

  const W     = 595.28;   // A4 width in points
  const LMARGIN = 50;
  const RMARGIN = 50;
  const CONTENT_W = W - LMARGIN - RMARGIN;

  // ── Brand colours
  const NAVY   = "#1e1b4b";
  const PURPLE = "#7c3aed";
  const TEAL   = "#0d9488";
  const RED    = "#dc2626";
  const AMBER  = "#d97706";
  const GREEN  = "#059669";
  const SLATE  = "#64748b";
  const LIGHT  = "#f8fafc";
  const BORDER = "#e2e8f0";

  function riskColor(pct) {
    if (pct >= 65) return RED;
    if (pct >= 35) return AMBER;
    return GREEN;
  }

  let y = 50;

  /* ════ HEADER BAND ════ */
  doc.rect(0, 0, W, 80).fill(NAVY);

  // Logo mark
  doc.roundedRect(LMARGIN, 18, 44, 44, 8).fill(PURPLE);
  doc.moveTo(LMARGIN + 10, 40).lineTo(LMARGIN + 17, 40)
     .lineTo(LMARGIN + 20, 32).lineTo(LMARGIN + 24, 52)
     .lineTo(LMARGIN + 28, 40).lineTo(LMARGIN + 34, 40)
     .strokeColor("white").lineWidth(2).stroke();

  // App name
  doc.fillColor("white").font("Helvetica-Bold").fontSize(20)
     .text("HeartGuard", LMARGIN + 54, 22);
  doc.fillColor("rgba(255,255,255,0.7)").font("Helvetica").fontSize(10)
     .text("AI-Powered Cardiac Risk Assessment Platform", LMARGIN + 54, 45);

  // Report label top-right
  doc.fillColor("white").font("Helvetica-Bold").fontSize(11)
     .text("CLINICAL ECG REPORT", W - RMARGIN - 160, 22, { width: 160, align: "right" });
  doc.fillColor("rgba(255,255,255,0.65)").font("Helvetica").fontSize(9)
     .text(`Generated: ${generatedAt}`, W - RMARGIN - 160, 40, { width: 160, align: "right" });
  doc.fillColor("rgba(255,255,255,0.65)").font("Helvetica").fontSize(9)
     .text(`Report ID: ${ecg.id?.slice(0, 8).toUpperCase() || "N/A"}`, W - RMARGIN - 160, 54, { width: 160, align: "right" });

  y = 96;

  /* ════ PATIENT + ECG INFO ROW ════ */
  // Left box — Patient Information
  doc.rect(LMARGIN, y, CONTENT_W / 2 - 6, 110).fill(LIGHT).stroke(BORDER);
  doc.fillColor(NAVY).font("Helvetica-Bold").fontSize(9)
     .text("PATIENT INFORMATION", LMARGIN + 12, y + 10);
  doc.moveTo(LMARGIN + 12, y + 22).lineTo(LMARGIN + CONTENT_W / 2 - 18, y + 22)
     .strokeColor(BORDER).lineWidth(1).stroke();

  const sexLabel = (profile.sex === 1 || profile.sex === "1") ? "Male" : "Female";
  const cpLabels = { 0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic" };
  const patientRows = [
    ["Age",          profile.age     ?? "N/A"],
    ["Sex",          sexLabel],
    ["Blood Pressure", `${profile.trestbps ?? "N/A"} mmHg`],
    ["Cholesterol",  `${profile.chol ?? "N/A"} mg/dL`],
    ["Chest Pain",   cpLabels[profile.cp] ?? "N/A"],
    ["Exercise Angina", profile.exang === 1 ? "Yes" : "No"],
    ["ST Depression (Oldpeak)", profile.oldpeak ?? "N/A"],
  ];
  let py = y + 28;
  patientRows.forEach(([k, v]) => {
    doc.fillColor(SLATE).font("Helvetica").fontSize(8).text(k + ":", LMARGIN + 12, py);
    doc.fillColor("#0f172a").font("Helvetica-Bold").fontSize(8).text(String(v), LMARGIN + 120, py);
    py += 11;
  });

  // Right box — ECG Recording Details
  const RX = LMARGIN + CONTENT_W / 2 + 6;
  doc.rect(RX, y, CONTENT_W / 2 - 6, 110).fill(LIGHT).stroke(BORDER);
  doc.fillColor(NAVY).font("Helvetica-Bold").fontSize(9)
     .text("ECG RECORDING DETAILS", RX + 12, y + 10);
  doc.moveTo(RX + 12, y + 22).lineTo(RX + CONTENT_W / 2 - 18, y + 22)
     .strokeColor(BORDER).lineWidth(1).stroke();

  const ecgRows = [
    ["Record ID",     ecg.id?.slice(0, 8).toUpperCase() || "N/A"],
    ["Recorded",      ecg.created_at ? new Date(ecg.created_at).toLocaleString("en-IN") : "N/A"],
    ["Heart Rate",    `${ecg.heart_rate ?? "N/A"} bpm`],
    ["Sample Rate",   `${ecg.sample_rate ?? "N/A"} Hz`],
    ["Duration",      `${ecg.duration_sec ?? "N/A"} seconds`],
    ["Total Samples", `${(ecg.samples || []).length}`],
    ["CNN Label",     ecg.label || "Unknown"],
  ];
  let ey = y + 28;
  ecgRows.forEach(([k, v]) => {
    doc.fillColor(SLATE).font("Helvetica").fontSize(8).text(k + ":", RX + 12, ey);
    doc.fillColor("#0f172a").font("Helvetica-Bold").fontSize(8).text(String(v), RX + 110, ey);
    ey += 11;
  });

  y += 120;

  /* ════ RISK SCORE BANNER ════ */
  const pct     = ml.risk_percentage ?? 0;
  const rc      = riskColor(pct);
  const rlabel  = pct >= 65 ? "HIGH RISK" : pct >= 35 ? "MODERATE RISK" : "LOW RISK";

  doc.rect(LMARGIN, y, CONTENT_W, 54).fill(rc + "18").stroke(rc + "55");

  // Big score
  doc.fillColor(rc).font("Helvetica-Bold").fontSize(36)
     .text(`${pct}%`, LMARGIN + 16, y + 9, { width: 80 });

  // Risk label + description
  doc.fillColor(rc).font("Helvetica-Bold").fontSize(13)
     .text(rlabel, LMARGIN + 96, y + 12);
  doc.fillColor(SLATE).font("Helvetica").fontSize(9)
     .text(`ML-computed cardiovascular disease probability based on ECG + clinical parameters. Source: ${ml.ml_source === "mock" ? "Heuristic model (ML service offline)" : "ML microservice"}.`,
       LMARGIN + 96, y + 29, { width: CONTENT_W - 110 });

  // Urgency tag
  const urgColors = { Routine: GREEN, Soon: AMBER, Urgent: "#ea580c", Emergency: RED };
  const urgBg     = urgColors[gemini.urgency] || SLATE;
  doc.rect(W - RMARGIN - 90, y + 10, 80, 22).fill(urgBg).roundedRect(W - RMARGIN - 90, y + 10, 80, 22, 4).fill(urgBg);
  doc.fillColor("white").font("Helvetica-Bold").fontSize(9)
     .text((gemini.urgency || "Routine").toUpperCase(), W - RMARGIN - 88, y + 17, { width: 76, align: "center" });

  y += 66;

  /* ════ ECG INTERPRETATION ════ */
  doc.rect(LMARGIN, y, CONTENT_W, 1).fill(BORDER);
  y += 8;
  doc.fillColor(PURPLE).font("Helvetica-Bold").fontSize(11).text("ECG INTERPRETATION", LMARGIN, y);
  y += 18;

  doc.rect(LMARGIN, y, 4, 50).fill(PURPLE);
  doc.fillColor("#0f172a").font("Helvetica").fontSize(9.5)
     .text(gemini.ecgInterpretation || "No interpretation available.", LMARGIN + 14, y, { width: CONTENT_W - 14, lineGap: 3 });
  y += 58;

  /* ════ AI SUMMARY ════ */
  doc.rect(LMARGIN, y, CONTENT_W, 1).fill(BORDER);
  y += 8;
  doc.fillColor(TEAL).font("Helvetica-Bold").fontSize(11).text("AI CLINICAL SUMMARY", LMARGIN, y);
  if (gemini.gemini_model) {
    doc.fillColor(SLATE).font("Helvetica").fontSize(8).text(`(${gemini.gemini_model})`, LMARGIN + 168, y + 2);
  }
  y += 18;

  doc.fillColor("#0f172a").font("Helvetica").fontSize(9.5)
     .text(gemini.summary || "No summary available.", LMARGIN, y, { width: CONTENT_W, lineGap: 3 });
  y += doc.heightOfString(gemini.summary || "", { width: CONTENT_W }) + 14;

  // Primary concern box
  if (gemini.primaryConcern) {
    doc.rect(LMARGIN, y, CONTENT_W, 28).fill("#fef9c3").stroke("#fde047");
    doc.fillColor("#713f12").font("Helvetica-Bold").fontSize(8.5)
       .text("⚠  PRIMARY CONCERN:", LMARGIN + 10, y + 9);
    doc.fillColor("#713f12").font("Helvetica").fontSize(8.5)
       .text(gemini.primaryConcern, LMARGIN + 120, y + 9, { width: CONTENT_W - 130 });
    y += 38;
  }

  /* ════ DISEASE BREAKDOWN ════ */
  doc.rect(LMARGIN, y, CONTENT_W, 1).fill(BORDER);
  y += 8;
  doc.fillColor(NAVY).font("Helvetica-Bold").fontSize(11).text("DISEASE-SPECIFIC RISK BREAKDOWN", LMARGIN, y);
  y += 18;

  const COLORS  = [PURPLE, "#2563eb", TEAL, AMBER, RED];
  const diseases = ml.disease_breakdown || [];
  const colW    = CONTENT_W / Math.min(diseases.length, 5);

  diseases.forEach((d, i) => {
    const cx = LMARGIN + i * colW;
    const barH = Math.round((d.pct / 100) * 50);
    // Bar
    doc.rect(cx + colW * 0.2, y + (50 - barH), colW * 0.6, barH).fill(COLORS[i] + "cc");
    // Pct label
    doc.fillColor(COLORS[i]).font("Helvetica-Bold").fontSize(10)
       .text(`${d.pct}%`, cx, y + 52, { width: colW, align: "center" });
    // Disease name (wrap)
    doc.fillColor(SLATE).font("Helvetica").fontSize(7)
       .text(d.name, cx + 2, y + 66, { width: colW - 4, align: "center" });
  });
  y += 90;

  /* ════ CLINICAL PARAMETERS TABLE ════ */
  doc.rect(LMARGIN, y, CONTENT_W, 1).fill(BORDER);
  y += 8;
  doc.fillColor(NAVY).font("Helvetica-Bold").fontSize(11).text("CLINICAL PARAMETERS", LMARGIN, y);
  y += 16;

  // Table header
  doc.rect(LMARGIN, y, CONTENT_W, 18).fill(NAVY);
  const colHeaders = ["Parameter", "Value", "Normal Range", "Status"];
  const colWidths  = [160, 100, 140, 95];
  let cx2 = LMARGIN + 8;
  colHeaders.forEach((h, i) => {
    doc.fillColor("white").font("Helvetica-Bold").fontSize(8).text(h, cx2, y + 5, { width: colWidths[i] });
    cx2 += colWidths[i];
  });
  y += 18;

  const paramRows = [
    ["Heart Rate (ECG)",    `${ecg.heart_rate ?? "N/A"} bpm`,       "60–100 bpm",     ecg.heart_rate >= 60 && ecg.heart_rate <= 100 ? "Normal" : "Abnormal"],
    ["Systolic BP",         `${profile.trestbps ?? "N/A"} mmHg`,    "90–120 mmHg",    profile.trestbps <= 120 ? "Normal" : profile.trestbps <= 140 ? "Elevated" : "High"],
    ["Cholesterol",         `${profile.chol ?? "N/A"} mg/dL`,       "<200 mg/dL",     profile.chol < 200 ? "Normal" : profile.chol < 240 ? "Borderline" : "High"],
    ["SpO2",                "97.5%",                                  "≥95%",           "Normal"],
    ["Temperature",         "37.0 °C",                               "36.1–37.2 °C",   "Normal"],
    ["ST Depression",       `${profile.oldpeak ?? "N/A"}`,           "0–1 mm",         (profile.oldpeak ?? 0) <= 1 ? "Normal" : "Elevated"],
    ["ECG Classification",  ecg.label || "Unknown",                  "Normal Sinus",   ecg.label?.toLowerCase() === "normal" ? "Normal" : "Review"],
  ];

  paramRows.forEach((row, ri) => {
    const rowBg = ri % 2 === 0 ? LIGHT : "white";
    doc.rect(LMARGIN, y, CONTENT_W, 16).fill(rowBg).stroke(BORDER);
    let px = LMARGIN + 8;
    row.forEach((cell, ci) => {
      const isStatus = ci === 3;
      const statusColor = cell === "Normal" ? GREEN : cell === "High" || cell === "Abnormal" || cell === "Elevated" ? RED : AMBER;
      if (isStatus) {
        doc.rect(px, y + 3, 70, 11).fill(statusColor + "22").roundedRect(px, y + 3, 70, 11, 3).fill(statusColor + "22");
        doc.fillColor(statusColor).font("Helvetica-Bold").fontSize(7.5).text(cell, px, y + 5, { width: 70, align: "center" });
      } else {
        doc.fillColor(ci === 0 ? SLATE : "#0f172a").font(ci === 0 ? "Helvetica" : "Helvetica-Bold").fontSize(8)
           .text(String(cell), px, y + 4, { width: colWidths[ci] - 8 });
      }
      px += colWidths[ci];
    });
    y += 16;
  });

  y += 12;

  /* ════ RECOMMENDATIONS ════ */
  // Check if we need a new page
  if (y > 680) { doc.addPage(); y = 50; }

  doc.rect(LMARGIN, y, CONTENT_W, 1).fill(BORDER);
  y += 8;

  // Two-column layout: Recommendations + Lifestyle
  const halfW = CONTENT_W / 2 - 8;

  // Left: Recommendations
  doc.fillColor(PURPLE).font("Helvetica-Bold").fontSize(11).text("CLINICAL RECOMMENDATIONS", LMARGIN, y);
  y += 16;
  (gemini.recommendations || []).forEach((rec, i) => {
    doc.circle(LMARGIN + 5, y + 4, 3).fill(PURPLE);
    doc.fillColor("#0f172a").font("Helvetica").fontSize(8.5)
       .text(rec, LMARGIN + 14, y, { width: halfW - 14, lineGap: 2 });
    y += doc.heightOfString(rec, { width: halfW - 14 }) + 8;
  });

  // Right: Lifestyle (positioned alongside recommendations — reset y for right col)
  const lifestyleY0 = y - ((gemini.recommendations || []).length * 22);
  let ly = lifestyleY0;

  doc.fillColor(TEAL).font("Helvetica-Bold").fontSize(11)
     .text("LIFESTYLE GUIDANCE", LMARGIN + halfW + 16, ly >= 50 ? ly : 50);
  ly = (ly >= 50 ? ly : 50) + 16;

  (gemini.lifestyle || []).forEach((tip) => {
    doc.circle(LMARGIN + halfW + 21, ly + 4, 3).fill(TEAL);
    doc.fillColor("#0f172a").font("Helvetica").fontSize(8.5)
       .text(tip, LMARGIN + halfW + 30, ly, { width: halfW - 14, lineGap: 2 });
    ly += doc.heightOfString(tip, { width: halfW - 14 }) + 8;
  });

  y = Math.max(y, ly) + 8;

  /* ════ FLAGGED VALUES ════ */
  const flagged = ml.flagged_features || [];
  if (flagged.length) {
    if (y > 720) { doc.addPage(); y = 50; }
    doc.rect(LMARGIN, y, CONTENT_W, 1).fill(BORDER);
    y += 8;
    doc.fillColor(RED).font("Helvetica-Bold").fontSize(11).text("FLAGGED ABNORMAL VALUES", LMARGIN, y);
    y += 14;
    flagged.forEach(f => {
      doc.rect(LMARGIN, y, CONTENT_W, 20).fill("#fef2f2").stroke("#fecaca");
      doc.fillColor(RED).font("Helvetica-Bold").fontSize(8.5)
         .text(`⚠  ${f.label}: ${f.value}`, LMARGIN + 10, y + 6);
      y += 24;
    });
    y += 4;
  }

  /* ════ FOOTER ════ */
  if (y > 730) { doc.addPage(); y = 50; }

  // Divider
  doc.rect(LMARGIN, y, CONTENT_W, 1).fill(BORDER);
  y += 10;

  // Signature block
  doc.rect(LMARGIN, y, 180, 50).fill(LIGHT).stroke(BORDER);
  doc.fillColor(SLATE).font("Helvetica").fontSize(8).text("Reviewed by:", LMARGIN + 10, y + 8);
  doc.fillColor("#0f172a").font("Helvetica-Bold").fontSize(9).text("HeartGuard AI System", LMARGIN + 10, y + 20);
  doc.fillColor(SLATE).font("Helvetica").fontSize(7).text("Cardiologist signature / stamp", LMARGIN + 10, y + 34);

  // Stamp area
  doc.rect(LMARGIN + 200, y, 80, 50).dash(3, { space: 3 }).stroke(BORDER).undash();
  doc.fillColor(SLATE).font("Helvetica").fontSize(7)
     .text("Doctor Stamp", LMARGIN + 200, y + 20, { width: 80, align: "center" });

  // Disclaimer
  doc.rect(LMARGIN + 310, y, CONTENT_W - 310, 50).fill("#fef2f2").stroke("#fecaca");
  doc.fillColor(RED).font("Helvetica-Bold").fontSize(7.5)
     .text("MEDICAL DISCLAIMER", LMARGIN + 320, y + 8);
  doc.fillColor("#7f1d1d").font("Helvetica").fontSize(7)
     .text(gemini.disclaimer || "This report is AI-generated and must not replace professional medical advice. Always consult a qualified cardiologist.",
       LMARGIN + 320, y + 20, { width: CONTENT_W - 330, lineGap: 2 });

  y += 60;

  // Bottom bar
  const BOTTOM = doc.page.height - 30;
  doc.rect(0, BOTTOM, W, 30).fill(NAVY);
  doc.fillColor("rgba(255,255,255,0.5)").font("Helvetica").fontSize(7)
     .text("HeartGuard — AI Cardiac Risk Assessment | Confidential Patient Report | Not for distribution without consent",
       LMARGIN, BOTTOM + 10, { width: W - 100, align: "center" });

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

/* POST /ecg/report/:user_id  — generate clinical PDF report */
router.post("/report/:user_id", async (req, res) => {
  const { user_id } = req.params;
  const { record_id } = req.body || {};
  console.log(`\n[ecg/report] Generating PDF for user: ${user_id}`);

  try {
    /* 1. Fetch ECG */
    let ecgQuery = supabase.from("ecg_data").select("*").eq("user_id", user_id);
    if (record_id) ecgQuery = ecgQuery.eq("id", record_id);
    else ecgQuery = ecgQuery.order("created_at", { ascending: false }).limit(1);

    const { data: ecgRows, error: ecgErr } = await ecgQuery;
    if (ecgErr || !ecgRows?.length)
      return res.status(404).json({ error: "No ECG record found." });
    const ecg = ecgRows[0];

    /* 2. Fetch profile */
    const { data: profileData } = await supabase
      .from("profile").select("age, sex, cp, trestbps, chol, restecg, exang, oldpeak")
      .eq("id", user_id).maybeSingle();
    const profile = profileData || {};

    /* 3. ML + Gemini */
    const payload = buildECGPayload(ecg, profile);
    const ml      = await mlPredict(payload);
    const gemini  = await geminiAnalyze(ecg, profile, ml);

    /* 4. Stream PDF */
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