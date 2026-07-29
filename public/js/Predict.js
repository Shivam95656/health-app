const user_id = localStorage.getItem("user_id");
if (!user_id) window.location.href = "login.html";

document.getElementById("sideEmail").textContent =
  localStorage.getItem("user_email") || "user@email.com";

let barChartInst = null, radarChartInst = null, donutChartInst = null, lineChartInst = null;

// ─────────────────────────────────────────────
// RISK HELPERS
// ─────────────────────────────────────────────
function riskColor(pct) {
  if (pct >= 65) return "#dc2626";
  if (pct >= 35) return "#d97706";
  return "#059669";
}
function riskLabel(pct) {
  if (pct >= 65) return ["high",   "High Risk"];
  if (pct >= 35) return ["medium", "Moderate Risk"];
  return               ["low",    "Low Risk"];
}

function safeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}
function clamp(value, min, max) {
  return Math.min(Math.max(safeNumber(value, min), min), max);
}

// ─────────────────────────────────────────────
// ML RISK SCORE  (logistic-style clinical weights)
// ─────────────────────────────────────────────
function computeRisk(profile, iot) {
  let score = 0;
  if      (profile.age > 60) score += 18;
  else if (profile.age > 50) score += 10;
  else if (profile.age > 40) score += 5;

  if (profile.sex == 1) score += 8;
  score += ({ "0": 15, "1": 8, "2": 5, "3": 0 }[profile.cp] || 0);

  if      (profile.trestbps > 160) score += 12;
  else if (profile.trestbps > 140) score += 7;
  else if (profile.trestbps > 120) score += 3;

  if      (profile.chol > 280) score += 10;
  else if (profile.chol > 240) score += 5;
  else if (profile.chol > 200) score += 2;

  score += ([10, 5, 10][profile.restecg] || 0);

  if      (profile.oldpeak > 3)   score += 12;
  else if (profile.oldpeak > 1.5) score += 7;
  else if (profile.oldpeak > 0.5) score += 3;

  if (profile.exang == 1) score += 10;

  if      (iot.thalach < 120) score += 15;
  else if (iot.thalach < 140) score += 8;
  else if (iot.thalach < 160) score += 3;

  if      (iot.spo2 < 90) score += 15;
  else if (iot.spo2 < 94) score += 8;
  else if (iot.spo2 < 97) score += 3;

  if (iot.temperature > 38.5 || iot.temperature < 36.0) score += 5;

  return Math.min(Math.round(score), 99);
}

// ─────────────────────────────────────────────
// DISEASE BREAKDOWN  (derived from risk + individual factors)
// Each disease gets an independent probability score
// ─────────────────────────────────────────────
function computeDiseaseBreakdown(profile, iot, overallRisk) {
  const base = overallRisk;

  // Coronary Artery Disease — driven by cholesterol, BP, age
  const cad = Math.min(Math.round(
    base * 0.4 +
    (profile.chol   > 240 ? 15 : profile.chol   > 200 ? 7 : 0) +
    (profile.trestbps > 140 ? 12 : profile.trestbps > 120 ? 5 : 0) +
    (profile.age    > 55  ? 10 : profile.age    > 45  ? 5 : 0)
  ), 95);

  // Hypertension — driven by BP, age, sex
  const htn = Math.min(Math.round(
    (profile.trestbps > 160 ? 70 : profile.trestbps > 140 ? 48 : profile.trestbps > 130 ? 28 : 15) +
    (profile.age > 60 ? 10 : profile.age > 50 ? 5 : 0) +
    (profile.sex == 1 ? 5 : 0)
  ), 95);

  // Arrhythmia — driven by ECG, exercise angina, heart rate
  const arr = Math.min(Math.round(
    (profile.restecg == 2 ? 55 : profile.restecg == 1 ? 35 : 10) +
    (profile.exang == 1 ? 15 : 0) +
    (iot.thalach < 120 ? 20 : iot.thalach < 140 ? 10 : 0) +
    (profile.oldpeak > 2 ? 10 : 0)
  ), 95);

  // Heart Failure — driven by overall risk, SpO2, oldpeak
  const hf = Math.min(Math.round(
    base * 0.35 +
    (iot.spo2 < 92 ? 20 : iot.spo2 < 95 ? 10 : 0) +
    (profile.oldpeak > 3 ? 15 : profile.oldpeak > 1.5 ? 8 : 0)
  ), 95);

  // Angina — driven by chest pain type, exercise angina
  const ang = Math.min(Math.round(
    ([70, 45, 25, 8][profile.cp] || 8) +
    (profile.exang == 1 ? 20 : 0) +
    (profile.oldpeak > 1.5 ? 10 : 0)
  ), 95);

  return { cad, htn, arr, hf, ang };
}

// ─────────────────────────────────────────────
// FACTOR CONTRIBUTIONS
// ─────────────────────────────────────────────
function getFactors(profile, iot) {
  const age = safeNumber(profile.age, 0);
  const cp = safeNumber(profile.cp, 0);
  const trestbps = safeNumber(profile.trestbps, 120);
  const chol = safeNumber(profile.chol, 180);
  const thalach = safeNumber(iot.thalach, 72);
  const spo2 = safeNumber(iot.spo2, 98);
  const oldpeak = safeNumber(profile.oldpeak, 0);
  const restecg = safeNumber(profile.restecg, 0);

  return [
    { name: "Age",            val: age > 55 ? 85 : age > 45 ? 55 : 25, color: "#3b82f6" },
    { name: "Chest Pain",     val: [60, 35, 20, 5][cp] || 5,                    color: "#f59e0b" },
    { name: "Blood Pressure", val: clamp(Math.round((trestbps - 80) / 1.4), 0, 99), color: "#ef4444" },
    { name: "Cholesterol",    val: clamp(Math.round((chol - 120) / 2.2), 0, 99),    color: "#8b5cf6" },
    { name: "Heart Rate",     val: clamp(Math.round((200 - thalach) / 1.5), 0, 99),      color: "#06b6d4" },
    { name: "SpO2",           val: clamp(Math.round((100 - spo2) * 5), 0, 99),            color: "#10b981" },
    { name: "ST Depression",  val: clamp(Math.round(oldpeak * 25), 0, 99),          color: "#f97316" },
    { name: "ECG Result",     val: clamp([10, 40, 75][restecg] || 10, 0, 99),                      color: "#ec4899" }
  ];
}

// ─────────────────────────────────────────────
// RENDER — FACTOR BARS
// ─────────────────────────────────────────────
function renderFactors(factors) {
  const el = document.getElementById("factorBars");
  const safeFactors = factors.map(f => {
    const raw = safeNumber(f.importance ?? f.value ?? 0, 0);
    return { ...f, val: clamp(raw, 0, 100) };
  });
  el.innerHTML = safeFactors.map(f => `
    <div class="factor-row">
      <span class="factor-name">${f.name}</span>
      <div class="factor-track">
        <div class="factor-fill" style="width:0%;background:${f.color};" data-target="${f.val}"></div>
      </div>
      <span class="factor-val">${f.val}%</span>
    </div>`).join("");
  setTimeout(() => {
    el.querySelectorAll(".factor-fill").forEach(b => { b.style.width = b.dataset.target + "%"; });
  }, 60);
}

// ─────────────────────────────────────────────
// RENDER — DISEASE DONUT CHART
// ─────────────────────────────────────────────
function renderDiseaseDonut(breakdown) {
  if (donutChartInst) donutChartInst.destroy();
  const ctx = document.getElementById("donutChart").getContext("2d");
  donutChartInst = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Coronary Artery Disease", "Hypertension", "Arrhythmia", "Heart Failure", "Angina"],
      datasets: [{
        data: [breakdown.cad, breakdown.htn, breakdown.arr, breakdown.hf, breakdown.ang],
        backgroundColor: ["#ef4444","#f59e0b","#8b5cf6","#3b82f6","#10b981"],
        borderWidth: 2,
        borderColor: "#ffffff",
        hoverOffset: 8
      }]
    },
    options: {
      responsive: true,
      cutout: "62%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { font: { family: "DM Sans", size: 11 }, boxWidth: 10, padding: 12 }
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed}%`
          }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────
// RENDER — BAR CHART (vitals vs normal)
// ─────────────────────────────────────────────
function renderBarChart(profile, iot) {
  if (barChartInst) barChartInst.destroy();
  const heartRate = clamp(safeNumber(iot.thalach, 0), 0, 220);
  const spo2 = clamp(safeNumber(iot.spo2, 0), 0, 100);
  const bp = clamp(safeNumber(profile.trestbps, 0), 0, 220);
  const chol = clamp(Math.round(safeNumber(profile.chol, 0) / 3), 0, 120);
  const stDepress = clamp(Math.round(safeNumber(profile.oldpeak, 0) * 15), 0, 100);

  barChartInst = new Chart(document.getElementById("barChart").getContext("2d"), {
    type: "bar",
    data: {
      labels: ["Heart Rate", "SpO2", "Blood Pressure", "Cholesterol", "ST Depress."],
      datasets: [
        {
          label: "Your Reading",
          data: [heartRate, spo2, bp, chol, stDepress],
          backgroundColor: ["#3b82f6","#10b981","#ef4444","#8b5cf6","#f97316"],
          borderRadius: 5, borderSkipped: false
        },
        {
          label: "Normal Max",
          data: [150, 100, 120, 67, 15],
          backgroundColor: "rgba(0,0,0,0.07)",
          borderRadius: 5, borderSkipped: false
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { font: { family: "DM Sans", size: 11 }, boxWidth: 10 } }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { family: "DM Sans", size: 11 } } },
        y: { grid: { color: "#f1f5f9" }, ticks: { font: { family: "DM Sans", size: 11 } } }
      }
    }
  });
}

// ─────────────────────────────────────────────
// RENDER — RADAR CHART
// ─────────────────────────────────────────────
function renderRadarChart(profile, iot) {
  if (radarChartInst) radarChartInst.destroy();
  const heartRate = clamp(Math.round(safeNumber(iot.thalach, 0) / 2.2), 0, 100);
  const spo2 = clamp(safeNumber(iot.spo2, 0), 0, 100);
  const temp = clamp(Math.round(safeNumber(iot.temperature, 36) * 2.5), 0, 100);
  const bp = clamp(Math.round(safeNumber(profile.trestbps, 120) / 2), 0, 100);
  const chol = clamp(Math.round(safeNumber(profile.chol, 180) / 6), 0, 100);
  const stDepress = clamp(Math.round(safeNumber(profile.oldpeak, 0) * 15), 0, 100);
  const ageFactor = clamp(Math.min(safeNumber(profile.age, 40), 80), 0, 80);

  radarChartInst = new Chart(document.getElementById("radarChart").getContext("2d"), {
    type: "radar",
    data: {
      labels: ["Heart Rate", "SpO2", "Temp", "BP", "Cholesterol", "ST Depress.", "Age Factor"],
      datasets: [{
        label: "Your Profile",
        data: [heartRate, spo2, temp, bp, chol, stDepress, ageFactor],
        backgroundColor: "rgba(29,78,216,0.1)",
        borderColor: "#1d4ed8",
        pointBackgroundColor: "#1d4ed8",
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          ticks: { display: false },
          grid: { color: "#e2e8f0" },
          pointLabels: { font: { family: "DM Sans", size: 11 } }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────
// RENDER — DISEASE PERCENTAGE TABLE CARDS
// ─────────────────────────────────────────────
function renderDiseaseCards(breakdown) {
  const diseases = [
    { name: "Coronary Artery Disease", abbr: "CAD", val: breakdown.cad, color: "#ef4444", bg: "#fef2f2", border: "#fecaca",
      desc: "Plaque buildup in coronary arteries restricting blood flow to the heart muscle." },
    { name: "Hypertension",            abbr: "HTN", val: breakdown.htn, color: "#f59e0b", bg: "#fffbeb", border: "#fde68a",
      desc: "Persistently elevated blood pressure putting strain on heart and blood vessels." },
    { name: "Arrhythmia",              abbr: "ARR", val: breakdown.arr, color: "#8b5cf6", bg: "#f5f3ff", border: "#ddd6fe",
      desc: "Irregular heartbeat pattern — too fast, too slow, or erratic rhythm." },
    { name: "Heart Failure",           abbr: "HF",  val: breakdown.hf,  color: "#3b82f6", bg: "#eff6ff", border: "#bfdbfe",
      desc: "Heart unable to pump enough blood to meet the body's needs efficiently." },
    { name: "Angina",                  abbr: "ANG", val: breakdown.ang, color: "#10b981", bg: "#ecfdf5", border: "#a7f3d0",
      desc: "Chest pain caused by reduced blood flow to the heart, often during exertion." }
  ];

  const el = document.getElementById("diseaseCards");
  el.innerHTML = diseases.map(d => {
    const [cls] = riskLabel(d.val);
    return `
    <div style="background:${d.bg};border:1px solid ${d.border};border-radius:12px;padding:1rem 1.25rem;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
        <div>
          <div style="font-size:13px;font-weight:600;color:${d.color};">${d.name}</div>
          <div style="font-size:11px;color:#64748b;margin-top:1px;">${d.desc}</div>
        </div>
        <div style="font-family:'DM Serif Display',serif;font-size:26px;color:${d.color};flex-shrink:0;margin-left:12px;">${d.val}%</div>
      </div>
      <div style="height:6px;background:rgba(0,0,0,0.08);border-radius:999px;overflow:hidden;">
        <div style="height:100%;width:0%;background:${d.color};border-radius:999px;transition:width 1s cubic-bezier(0.16,1,0.3,1);" data-target="${d.val}" class="disease-bar"></div>
      </div>
    </div>`;
  }).join("");

  setTimeout(() => {
    el.querySelectorAll(".disease-bar").forEach(b => { b.style.width = b.dataset.target + "%"; });
  }, 80);
}

// ─────────────────────────────────────────────
// GAUGE
// ─────────────────────────────────────────────
function updateGauge(pct) {
  const path  = document.getElementById("gaugePath");
  const total = 220;
  path.style.stroke = riskColor(pct);
  setTimeout(() => { path.style.strokeDashoffset = total - (total * pct / 100); }, 60);
  document.getElementById("gaugeText").textContent  = pct + "%";
  document.getElementById("gaugeLabel").textContent = riskLabel(pct)[1];

  const [cls, lbl] = riskLabel(pct);
  document.getElementById("riskBadge").innerHTML =
    `<span class="risk-badge ${cls}"><span class="risk-dot"></span>${lbl}</span>`;

  const sc = document.getElementById("statRisk");
  sc.className = `stat-card risk-${cls === "high" ? "high" : cls === "medium" ? "mid" : "low"}`;
  document.getElementById("statRiskVal").textContent  = pct + "%";
  document.getElementById("statRiskDesc").textContent = lbl;
}

// ─────────────────────────────────────────────
// GEMINI AI INSIGHTS
// ─────────────────────────────────────────────
async function fetchGeminiInsights(profile, iot, risk, breakdown) {
  const panel = document.getElementById("geminiPanel");
  panel.innerHTML = `
    <div class="ai-thinking">
      <div class="ai-dots"><span></span><span></span><span></span></div>
      Gemini is analysing your health data...
    </div>`;

  try {
    const res  = await fetch("/gemini", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ profile, iot, risk, breakdown })
    });
    const data = await res.json();
    const text = data.text || data.error || "Could not fetch AI insights.";

    const sections = [
      { key: "SUMMARY",                   icon: "M12 22s8-4 8-10V5l-8-2-8 2v7c0 6 8 10 8 10z" },
      { key: "KEY CONCERNS",              icon: "M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" },
      { key: "LIFESTYLE RECOMMENDATIONS", icon: "M22 11.08V12a10 10 0 1 1-5.93-9.14" },
      { key: "WHEN TO SEE A DOCTOR",      icon: "M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0-2-2z" }
    ];

    let html = "";
    sections.forEach((sec, i) => {
      const nextKey = sections[i + 1] ? sections[i + 1].key : null;
      const start   = text.indexOf(sec.key);
      if (start === -1) return;
      const end     = nextKey ? text.indexOf(nextKey) : text.length;
      const content = text.substring(start + sec.key.length, end > -1 ? end : undefined)
                          .replace(/^[\s:\-]+/, "").trim();
      html += `
        <div class="ai-section">
          <div class="ai-section-title">
            <svg viewBox="0 0 24 24" style="width:12px;height:12px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;vertical-align:middle;margin-right:5px;">
              <path d="${sec.icon}"/>
            </svg>${sec.key}
          </div>
          <div class="ai-text">${content}</div>
        </div>`;
    });

    panel.innerHTML = html || `<div class="ai-text">${text}</div>`;
  } catch (e) {
    panel.innerHTML = `<div style="color:var(--danger);font-size:13px;">Could not reach Gemini API. Check your GEMINI_API_KEY in .env</div>`;
  }
}

// ─────────────────────────────────────────────
// HISTORY TABLE
// ─────────────────────────────────────────────
async function loadHistory() {
  const panel = document.getElementById("historyPanel");
  try {
    const res  = await fetch(`/iot/${user_id}`);
    const rows = await res.json();

    if (!rows || rows.error || rows.length === 0) {
      panel.innerHTML = `<div class="placeholder">No IoT reading history yet.</div>`;
      return;
    }

    panel.innerHTML = `
      <table class="history-table">
        <thead>
          <tr>
            <th>Date &amp; Time</th>
            <th>Heart Rate</th>
            <th>SpO2</th>
            <th>Temperature</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(r => `
            <tr>
              <td>${new Date(r.created_at).toLocaleString("en-IN", { day:"2-digit", month:"short", year:"numeric", hour:"2-digit", minute:"2-digit" })}</td>
              <td>${r.thalach     ?? "—"} bpm</td>
              <td>${r.spo2        ?? "—"}%</td>
              <td>${r.temperature ?? "—"}°C</td>
            </tr>`).join("")}
        </tbody>
      </table>`;
  } catch (e) {
    panel.innerHTML = `<div class="placeholder">Could not load history.</div>`;
  }
}

// ─────────────────────────────────────────────
// SHOW IoT CARD (latest reading display)
// ─────────────────────────────────────────────
function showIotCard(iot, timestamp) {
  const card = document.getElementById("iotDataCard");
  if (!card) return;

  const time = new Date(timestamp).toLocaleString("en-IN", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit"
  });

  card.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
      <span style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);">
        Latest IoT Reading
      </span>
      <span style="font-size:11px;color:var(--text-muted);">${time}</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
      <div style="text-align:center;background:var(--slate-light);border:1px solid var(--border);border-radius:9px;padding:14px 10px;">
        <div style="font-size:22px;font-weight:700;color:var(--text);">${iot.thalach}</div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:3px;">bpm — Heart Rate</div>
      </div>
      <div style="text-align:center;background:var(--slate-light);border:1px solid var(--border);border-radius:9px;padding:14px 10px;">
        <div style="font-size:22px;font-weight:700;color:var(--text);">${iot.temperature}°C</div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:3px;">Temperature</div>
      </div>
      <div style="text-align:center;background:var(--slate-light);border:1px solid var(--border);border-radius:9px;padding:14px 10px;">
        <div style="font-size:22px;font-weight:700;color:var(--text);">${iot.spo2}%</div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:3px;">SpO2</div>
      </div>
    </div>`;
}

// ─────────────────────────────────────────────
// MAIN — RUN PREDICTION
// Fetches profile + latest IoT from DB, computes everything
// ─────────────────────────────────────────────
async function runPrediction() {
  const alertEl = document.getElementById("inputAlert");
  const btn     = document.getElementById("predictBtn");
  const runBtn  = document.getElementById("runBtn");

  alertEl.classList.remove("show");
  btn.textContent    = "Fetching data...";
  if (runBtn) runBtn.textContent = "Fetching data...";

  // 1. Fetch profile from DB
  let profile;
  try {
    const res = await fetch(`/profile/${user_id}`);
    profile   = await res.json();
    if (!profile || !profile.age) throw new Error("no profile");
  } catch (e) {
    alertEl.textContent = "Please complete your Medical Profile before running a prediction.";
    alertEl.classList.add("show");
    resetBtnText(); return;
  }

  // 2. Fetch latest IoT from DB
  let iot;
  try {
    const res  = await fetch(`/iot/${user_id}`);
    const rows = await res.json();
    if (!rows || rows.error || rows.length === 0) throw new Error("no iot");
    const latest = rows[0];
    iot = {
      thalach:     parseFloat(latest.thalach),
      temperature: parseFloat(latest.temperature),
      spo2:        parseFloat(latest.spo2)
    };
    showIotCard(iot, latest.created_at);
  } catch (e) {
    alertEl.textContent = "No IoT data found. Please send a sensor reading from your ESP32 device first.";
    alertEl.classList.add("show");
    resetBtnText(); return;
  }

  // 3. Compute scores
  const risk      = computeRisk(profile, iot);
  const breakdown = computeDiseaseBreakdown(profile, iot, risk);

  // 4. Show results section
  document.getElementById("initialPlaceholder").style.display = "none";
  document.getElementById("resultsSection").style.display     = "block";

  // 5. Stats
  document.getElementById("statThalach").textContent = iot.thalach     + " bpm";
  document.getElementById("statSpo2").textContent    = iot.spo2        + "%";
  document.getElementById("statTemp").textContent    = iot.temperature + "°C";

  // 6. Render all visuals
  updateGauge(risk);
  renderFactors(getFactors(profile, iot));
  renderBarChart(profile, iot);
  renderRadarChart(profile, iot);
  renderDiseaseDonut(breakdown);
  renderDiseaseCards(breakdown);
  loadHistory();

  // 7. Gemini (async — non-blocking)
  fetchGeminiInsights(profile, iot, risk, breakdown);

  resetBtnText();
}

function resetBtnText() {
  const icon = `<svg viewBox="0 0 24 24" style="width:15px;height:15px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
  const btn    = document.getElementById("predictBtn");
  const runBtn = document.getElementById("runBtn");
  if (btn)    btn.innerHTML    = `${icon} Predict`;
  if (runBtn) runBtn.innerHTML = `${icon} Run Prediction`;
}

function signOut() {
  localStorage.clear();
  window.location.href = "login.html";
}

// ─────────────────────────────────────────────
// ON PAGE LOAD — auto-show latest IoT + history
// ─────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
  try {
    const res  = await fetch(`/iot/${user_id}`);
    const rows = await res.json();
    if (rows && rows.length > 0) {
      showIotCard(
        { thalach: rows[0].thalach, temperature: rows[0].temperature, spo2: rows[0].spo2 },
        rows[0].created_at
      );
    }
  } catch (_) {}
  loadHistory();
});