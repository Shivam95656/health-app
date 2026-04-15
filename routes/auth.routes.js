const express  = require("express");
const router   = express.Router();
const supabase = require("../supabase");

// POST /register
router.post("/register", async (req, res) => {
  const { email, password } = req.body;

  const { error } = await supabase.auth.signUp({ email, password });

  if (error) return res.json({ error: error.message });
  res.json({ message: "Registered successfully" });
});

// POST /login
router.post("/login", async (req, res) => {
  const { email, password } = req.body;

  const { data, error } = await supabase.auth.signInWithPassword({ email, password });

  if (error) return res.json({ error: error.message });
  res.json({ user: data.user });
});

module.exports = router;