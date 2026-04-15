# HeartGuard Connection Setup & Troubleshooting

## Quick Status Check

Run this to test all connections:

```bash
npm start
# Server should start at http://localhost:3000
```

Then visit: http://localhost:3000/test
- Should see: ✅ "Server working"

---

## Connection Dependencies

### 1. **Supabase Database** ✓ (CONFIGURED)
- **Status**: Connected ✅
- **URL**: https://epsjghqhagdworfvmrto.supabase.co
- **Required Tables**:
  - `auth.users` (auto-created by Supabase)
  - `public.profile` (for user health data)
  - `public.iot_data` (for sensor readings)
  - `public.final_data` (for merged predictions)

**Test**: 
```bash
curl http://localhost:3000/test
# Should respond: "Server working"
```

---

### 2. **ML Service** ⚠️ (NEEDS TO START)
- **Status**: **NOT RUNNING** - Must be started manually
- **URL**: http://localhost:5001
- **Endpoint**: POST `/predict`
- **Source**: `ml_service/predict_service.py`

**Start ML Service**:
```bash
cd ml_service
python predict_service.py
# Should show: "Listening on port 5001"
```

**Test**:
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 50, "sex": 1, "trestbps": 120}'
# Should respond with risk prediction
```

---

### 3. **Gemini API** ✓ (CONFIGURED)
- **Status**: Connected ✅
- **API Key**: AIzaSyA1NhRcH3uWaO-ePlZo0vQVi3S6difdX_I
- **Endpoint**: https://generativelanguage.googleapis.com

**Test**: 
```bash
# Make a prediction to trigger Gemini analysis
curl -X GET http://localhost:3000/cardio/latest
# Will call Gemini for clinical interpretation
```

---

## Connection Flow

```
Frontend (index.html)
    ↓
Express Server (port 3000)
    ├─→ /auth → Supabase Auth
    ├─→ /profile → Supabase (profile table)
    ├─→ /iot → Supabase (iot_data table) 
    ├─→ /final → Supabase (final_data table)
    ├─→ /cardio → ML Service + Gemini
    │   ├─→ localhost:5001/predict (ML)
    │   └─→ generativelanguage.googleapis.com (Gemini)
    └─→ /gemini → Gemini API directly
```

---

## Required Supabase Tables

### 1. **profile** table
```sql
CREATE TABLE profile (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text UNIQUE NOT NULL,
  age integer,
  sex integer,
  cp integer,
  trestbps integer,
  chol integer,
  restecg integer,
  oldpeak float8,
  exang integer,
  created_at timestamp DEFAULT now()
);
```

### 2. **iot_data** table
```sql
CREATE TABLE iot_data (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  thalach integer,
  temperature float8,
  spo2 float8,
  created_at timestamp DEFAULT now()
);
```

### 3. **final_data** table
```sql
CREATE TABLE final_data (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  age integer,
  sex integer,
  cp integer,
  trestbps integer,
  chol integer,
  restecg integer,
  oldpeak float8,
  exang integer,
  thalach integer,
  temperature float8,
  spo2 float8,
  prediction float8,
  created_at timestamp DEFAULT now()
);
```

---

## Node.js Startup Checklist

✅ `npm install` - All dependencies installed
✅ `.env` file - ML_SERVICE_URL added
✅ `server.js` - Routes properly mounted
✅ `/routes` - All route files export properly
✅ `supabase.js` - Singleton client created

---

## Common Connection Issues & Fixes

### ❌ Issue: "Cannot GET /profile/:user_id"
**Fix**: Make sure routes are mounted correctly in `server.js`:
```javascript
app.use("/profile", require("./routes/profile.routes"));
app.use("/iot",     require("./routes/iot.routes"));
app.use("/final",   require("./routes/final.routes"));
app.use("/gemini",  require("./routes/gemini.routes"));
app.use("/cardio",  require("./routes/cardio.routes"));
```

---

### ❌ Issue: "Supabase connection timeout"
**Fix**: Check .env file has credentials:
```bash
# In .env:
SUPABASE_URL=https://epsjghqhagdworfvmrto.supabase.co
SUPABASE_KEY=eyJhbGc...
```

---

### ❌ Issue: "ML service error: 500" or "ML service error: Connection refused"
**Fix**: Start ML service in separate terminal:
```bash
cd ml_service
python predict_service.py
```

If Python not installed:
```bash
pip install -r ml_service/requirements.txt
```

---

### ❌ Issue: "Gemini API error"
**Fix**: Verify API key is active at: https://console.cloud.google.com

---

### ❌ Issue: CORS errors in browser console
**Fix**: server.js already has CORS enabled:
```javascript
app.use(cors());
```

If still getting CORS errors, update to:
```javascript
app.use(cors({
  origin: "http://localhost:3000",
  credentials: true
}));
```

---

## Complete Startup Procedure

**Terminal 1 - Node Server**:
```bash
cd d:\project\health-app
npm install
npm start
# Runs on http://localhost:3000
```

**Terminal 2 - ML Service**:
```bash
cd d:\project\health-app\ml_service
python predict_service.py
# Runs on http://localhost:5001
```

**Terminal 3 - Open Browser**:
```
http://localhost:3000
```

---

## Testing All Connections

### 1. Server Health
```bash
curl http://localhost:3000/test
→ "Server working"
```

### 2. Supabase Connection
```bash
curl -X POST http://localhost:3000/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "test123"}'
# Should respond with error or user data
```

### 3. ML Service
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 50, "sex": 1, "trestbps": 120, "chol": 200}'
# Should return risk percentage
```

### 4. Frontend to Backend
Open http://localhost:3000 and check browser console:
- Should load without CORS errors
- Sign in should work
- Profile save should work
- Predictions should work

---

## Environment Variables Reference

```
SUPABASE_URL=https://epsjghqhagdworfvmrto.supabase.co
SUPABASE_KEY=eyJhbGc...                           # Can be found in Supabase dashboard
ML_SERVICE_URL=http://localhost:5001              # ML microservice
GEMINI_API_KEY=AIzaSyA1NhRcH3uWaO-ePlZo0vQVi3S6difdX_I  # Google Gemini API
PORT=3000                                         # Express server port
```

---

## Production vs Development

### Development (Current)
- Server: http://localhost:3000
- ML: http://localhost:5001
- Database: Remote Supabase instance

### Production (Later)
- Server: Docker container
- ML: Docker container
- Database: Supabase (same)
- API: Cloud deployment (.vercel, .railway, .render, etc.)

---

## Next Steps

1. ✅ Verify `.env` file has `ML_SERVICE_URL`
2. ✅ Start Node server: `npm start`
3. ✅ Start ML service: `python ml_service/predict_service.py`
4. ✅ Open http://localhost:3000
5. ✅ Test login/register flow
6. ✅ Test prediction functionality

If any connection fails, check the relevant section above!
