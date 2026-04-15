const express  = require("express");
const router   = express.Router();
const supabase = require("../supabase");

// POST /iot  — save IoT reading and merge into final_data
router.post("/", async (req, res) => {
  const { user_id, thalach, temperature, spo2 } = req.body;

  if (!user_id)                          return res.json({ error: "User ID is required." });
  if (!thalach || !temperature || !spo2) return res.json({ error: "All IoT fields are required." });

  const iotValues = {
    thalach:     parseFloat(thalach),
    temperature: parseFloat(temperature),
    spo2:        parseFloat(spo2)
  };

  // STEP 1 — Save raw IoT reading
  const { error: iotError } = await supabase
    .from("iot_data")
    .insert([{ user_id, ...iotValues }]);

  if (iotError) return res.json({ error: iotError.message });

  // STEP 2 — Fetch user profile
  const { data: profile, error: profileError } = await supabase
    .from("profile")
    .select("age, sex, cp, trestbps, chol, restecg, oldpeak, exang")
    .eq("id", user_id)
    .single();

  if (profileError || !profile) {
    return res.json({
      message: "IoT data saved. No profile found — final_data not updated.",
      warning: true
    });
  }

  // STEP 3 — Insert merged row into final_data
  const { error: finalError } = await supabase
    .from("final_data")
    .insert([{
      user_id,
      age:      profile.age,
      sex:      profile.sex,
      cp:       profile.cp,
      trestbps: profile.trestbps,
      chol:     profile.chol,
      restecg:  profile.restecg,
      oldpeak:  profile.oldpeak,
      exang:    profile.exang,
      ...iotValues,
      prediction: null
    }]);

  if (finalError) return res.json({ error: finalError.message });

  res.json({ message: "IoT data saved and merged into final_data successfully." });
});

// GET /iot/:user_id  — last 10 IoT readings
router.get("/:user_id", async (req, res) => {
  const { user_id } = req.params;

  const { data, error } = await supabase
    .from("iot_data")
    .select("*")
    .eq("user_id", user_id)
    .order("created_at", { ascending: false })
    .limit(10);

  if (error) return res.json({ error: error.message });
  res.json(data);
});

module.exports = router;