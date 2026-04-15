const express  = require("express");
const router   = express.Router();
const supabase = require("../supabase");

// GET /final/:user_id  — prediction history (last 20)
router.get("/:user_id", async (req, res) => {
  const { user_id } = req.params;

  const { data, error } = await supabase
    .from("final_data")
    .select("*")
    .eq("user_id", user_id)
    .order("created_at", { ascending: false })
    .limit(20);

  if (error) return res.json({ error: error.message });
  res.json(data);
});

module.exports = router;