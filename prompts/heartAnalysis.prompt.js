/**
 * heartAnalysis.prompt.js
 * Builds the Gemini prompt for heart disease risk analysis.
 * Includes individual disease breakdown percentages.
 */

const CP_LABELS      = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"];
const RESTECG_LABELS = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"];

function buildPrompt(profile, iot, risk, breakdown) {
  const sex       = profile.sex == 1 ? "Male" : "Female";
  const exang     = profile.exang == 1 ? "Yes" : "No";
  const cpType    = CP_LABELS[profile.cp]          || "Unknown";
  const ecgResult = RESTECG_LABELS[profile.restecg] || "Unknown";
  const riskLevel = risk >= 65 ? "HIGH" : risk >= 35 ? "MODERATE" : "LOW";

  const bd = breakdown || {};

  return `
You are a clinical AI assistant specialising in cardiovascular risk assessment.
A patient's data has been collected from their medical profile and IoT wearable device.

────────────────────────────────
PATIENT PROFILE
────────────────────────────────
Age             : ${profile.age} years
Sex             : ${sex}
Blood Pressure  : ${profile.trestbps} mmHg (resting)
Cholesterol     : ${profile.chol} mg/dl (serum)
Chest Pain Type : ${cpType} (Type ${profile.cp})
Resting ECG     : ${ecgResult}
ST Depression   : ${profile.oldpeak} mm (oldpeak)
Exercise Angina : ${exang}

────────────────────────────────
REAL-TIME IoT VITALS
────────────────────────────────
Max Heart Rate  : ${iot.thalach} bpm
SpO2            : ${iot.spo2}%
Body Temp       : ${iot.temperature}°C

────────────────────────────────
ML RISK SCORES
────────────────────────────────
Overall Risk              : ${risk}% (${riskLevel})
Coronary Artery Disease   : ${bd.cad  ?? "N/A"}%
Hypertension              : ${bd.htn  ?? "N/A"}%
Arrhythmia                : ${bd.arr  ?? "N/A"}%
Heart Failure             : ${bd.hf   ?? "N/A"}%
Angina                    : ${bd.ang  ?? "N/A"}%

────────────────────────────────

Respond ONLY using the following four labelled sections.
Do NOT use markdown symbols like *, #, or -.
Write in plain, clear English that a non-medical person can understand.

SUMMARY:
Write 3-4 sentences summarising the patient's overall cardiovascular health. Mention their top 2 disease risks by name and what the scores mean.

KEY CONCERNS:
List exactly 3 specific risk factors from this patient's data. Number them 1, 2, 3. Be specific (e.g. mention actual values like BP 148 mmHg).

LIFESTYLE RECOMMENDATIONS:
Give exactly 3 specific, actionable steps. Number them 1, 2, 3. Tailor them to this patient's highest risk factors.

WHEN TO SEE A DOCTOR:
Based on the overall risk of ${risk}% (${riskLevel}) and the disease breakdown, give clear guidance on urgency. Mention which specialist they should see if needed.
`.trim();
}

module.exports = buildPrompt;