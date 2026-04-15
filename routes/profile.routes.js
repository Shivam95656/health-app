const express  = require("express");
const router   = express.Router();
const supabase = require("../supabase");

// POST /profile  — save / update profile
router.post("/", async (req, res) => {
  const { user_id, age, sex, cp, trestbps, chol, restecg, oldpeak, exang } = req.body;

  const { error } = await supabase.from("profile").upsert([
    { id: user_id, age, sex, cp, trestbps, chol, restecg, oldpeak, exang }
  ]);

  if (error) return res.json({ error: error.message });
  res.json({ message: "Profile saved successfully" });
});

// GET /profile/:user_id  — fetch profile
router.get("/:user_id", async (req, res) => {
  const { user_id } = req.params;

  const { data, error } = await supabase
    .from("profile")
    .select("*")
    .eq("id", user_id)
    .single();

  if (error) return res.json({ error: error.message });
  res.json(data);
});

module.exports = router;