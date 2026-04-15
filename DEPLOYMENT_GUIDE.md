# HeartGuard Deployment & Team Collaboration Guide

## Quick Start: Local Development (Team)

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/health-app.git
cd health-app
```

### Step 2: Install Dependencies
```powershell
# Backend
npm install

# ML Service
cd ml_service
pip install -r requirements.txt
cd ..
```

### Step 3: Start All Services

**Terminal 1 - Backend Server:**
```powershell
npm start
# Runs on http://localhost:3000
```

**Terminal 2 - ML Service:**
```powershell
cd ml_service
python predict_service.py
# Runs on http://localhost:5001
```

### Step 4: Access Application
```
http://localhost:3000/cardio.html
```

---

## GitHub Setup for Group Editing

### Initial Setup (Project Owner)

**1. Create Repository on GitHub**
- Go to github.com → "New" button
- Name: `health-app`
- Private or Public (choose)
- Click "Create repository"

**2. Initialize Local Git**
```powershell
cd d:\project\health-app
git init
git add .
git commit -m "Initial commit: HeartGuard health app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/health-app.git
git push -u origin main
```

**3. Add Team Members**
- Go to GitHub repo → Settings → Collaborators
- Click "Add people"
- Enter each team member's GitHub username
- Give permissions: "Write" for editors, "Read" for reviewers

---

### For Team Members: Daily Workflow

**Start of day - Get latest code:**
```powershell
git pull origin main
```

**Make changes - commit and push:**
```powershell
# After editing files
git add .
git commit -m "Descriptive message about changes"
git push origin main
```

**Creating a feature branch (recommended):**
```powershell
# Create new branch
git checkout -b feature/your-feature-name

# Make changes
git add .
git commit -m "Your changes"

# Push to GitHub
git push origin feature/your-feature-name

# On GitHub: Create Pull Request for review
# After approval: Merge to main
```

**To update your code after pull request merge:**
```powershell
git checkout main
git pull origin main
```

---

## Deployment Options

### Option 1: Local Network Deployment (Team Development)
**Best for:** Testing with other team members on same network

**Your IP Address:**
```powershell
ipconfig
# Look for "IPv4 Address" under your network adapter
# Example: 192.168.x.x or 10.0.x.x
```

**Team members access:**
```
http://YOUR_IP:3000/cardio.html
```

**Make sure firewall allows port 3000:**
```powershell
# Windows Firewall - Add Rule
New-NetFirewallRule -DisplayName "Port 3000" -Direction Inbound -LocalPort 3000 -Protocol TCP -Action Allow
```

---

### Option 2: Render.com Deployment (Recommended - FREE)
**Best for:** First cloud deployment, fast setup

**Step 1: Sign Up**
- Go to render.com
- Sign up with GitHub account

**Step 2: Deploy Backend**
- Click "New" → "Web Service"
- Select GitHub repo
- Build command: `npm install`
- Start command: `npm start`
- Set environment variables (see below)
- Deploy

**Step 3: Deploy ML Service**
- Click "New" → "Web Service"
- Same repo
- Build command: `pip install -r ml_service/requirements.txt`
- Start command: `cd ml_service && python predict_service.py`
- Deploy

**Step 4: Update Backend URL**
In `server.js`, update ML service URL:
```javascript
const ML_SERVICE_URL = "https://your-ml-service.onrender.com";
```

---

### Option 3: Heroku Deployment (Easy - PAID)
**Step 1: Install Heroku CLI**
```powershell
# Download from heroku.com/download
heroku login
```

**Step 2: Deploy Backend**
```powershell
cd d:\project\health-app
heroku create your-app-name
heroku config:set SUPABASE_URL=your_url
heroku config:set GEMINI_API_KEY=your_key
git push heroku main
```

**Step 3: Check Logs**
```powershell
heroku logs --tail
```

---

## Environment Variables Setup

### Create `.env` File (NEVER push to GitHub)
```
# Supabase Database
SUPABASE_URL=https://epsjghqhagdworfvmrto.supabase.co
SUPABASE_KEY=your_supabase_key_here

# Gemini API
GEMINI_API_KEY=AIzaSyA1NhRcH3uWaO-ePlZo0vQVi3S6difdX_I

# Port
PORT=3000
```

### Create `.gitignore` (Prevent pushing secrets)
```
.env
.env.local
.env.*.local
node_modules/
*.pyc
__pycache__/
.DS_Store
dist/
build/
```

---

## Production Checklist

- [ ] All environment variables set in deployment platform
- [ ] ML service deployed separately
- [ ] Supabase tables created (see CONNECTION_SETUP.md)
- [ ] CORS enabled for frontend domain
- [ ] SSL/HTTPS enabled
- [ ] Email notifications configured
- [ ] Database backups configured
- [ ] Error logging enabled
- [ ] Team members added to GitHub repo
- [ ] Protected main branch (Settings → Branches → Add rule)

---

## Troubleshooting

**"Cannot connect to ML service"**
```
→ Check if predict_service.py is running
→ Verify ML_SERVICE_URL in server.js
→ Check firewall allows port 5001
```

**"Supabase connection error"**
```
→ Check .env file has correct SUPABASE_URL and SUPABASE_KEY
→ Verify internet connection
→ Check Supabase dashboard status
```

**"Port 3000 already in use"**
```powershell
# Find process using port 3000
netstat -ano | findstr :3000

# Kill process (replace PID with actual number)
taskkill /PID PID_NUMBER /F

# Or use different port
PORT=3001 npm start
```

**"Git push rejected"**
```powershell
# Make sure you're on main branch
git checkout main
git pull origin main
git push origin main
```

---

## Useful Commands

**Check all services running:**
```powershell
# Backend
curl http://localhost:3000/test

# ML Service  
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d "{\"age\":50,\"sex\":1,\"trestbps\":120}"
```

**View Git log:**
```powershell
git log --oneline
```

**See who changed what:**
```powershell
git blame filename.js
```

**Undo last commit (not pushed):**
```powershell
git reset HEAD~1
```

**See all branches:**
```powershell
git branch -a
```

---

## Contact & Support

For issues:
1. Check CONNECTION_SETUP.md for service errors
2. Create GitHub Issue for bugs
3. Use code reviews on Pull Requests for team feedback
