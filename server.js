require("dotenv").config();

const express = require("express");
const cors    = require("cors");

const app = express();

app.use(cors());
app.use(express.json());
app.use(express.static("public"));

// ── ROUTES ──
app.use("/",        require("./routes/auth.routes"));
app.use("/profile", require("./routes/profile.routes"));
app.use("/iot",     require("./routes/iot.routes"));
app.use("/final",   require("./routes/final.routes"));
app.use("/gemini",  require("./routes/gemini.routes"));
app.use("/cardio", require("./routes/cardio.routes"));
app.use("/ecg",    require("./routes/ecg.routes"));
 



app.get("/test", (req, res) => res.send("Server working"));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));