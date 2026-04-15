
const express     = require("express");
const router      = express.Router();
const buildPrompt = require("../prompts/heartAnalysis.prompt");

// POST /gemini
router.post("/", async (req, res) => {
  const { profile, iot, risk, breakdown } = req.body;

  if (!profile || !iot || risk === undefined) {
    return res.json({ error: "profile, iot, and risk are required." });
  }

  const prompt = buildPrompt(profile, iot, risk, breakdown);

   try {
     const response = await fetch(
      `https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }]
        })
      }
    );


    const data = await response.json();
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text
               || "No response from Gemini.";

    res.json({ text });
  } catch (e) {
    res.json({ error: "Gemini API error: " + e.message });
  }
});

module.exports = router;


