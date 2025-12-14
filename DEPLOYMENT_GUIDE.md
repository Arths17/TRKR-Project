# 🚀 GitHub Deployment Guide

## Quick Deploy (Automated)

```bash
./deploy_github.sh
```

This script will:
1. Initialize git repository (if needed)
2. Stage all files
3. Commit changes
4. Prompt for GitHub repository URL
5. Push to GitHub

---

## Manual Deployment Steps

### 1️⃣ Initialize Git Repository

```bash
git init
```

### 2️⃣ Configure Git (First Time Only)

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 3️⃣ Stage Files

```bash
# Add all files
git add .

# Or add specific files
git add f1_tracker_app.py requirements.txt README_GITHUB.md
```

### 4️⃣ Commit Changes

```bash
git commit -m "Initial commit: F1 Prediction Tracker"
```

### 5️⃣ Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click **New Repository**
3. Name: `f1predict` (or your choice)
4. Description: "🏎️ F1 Race Prediction System with ML & Interactive Dashboard"
5. **Do NOT** initialize with README (we already have one)
6. Click **Create repository**

### 6️⃣ Connect to GitHub

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/f1predict.git

# Verify remote
git remote -v
```

### 7️⃣ Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

---

## 📦 What Gets Uploaded

### ✅ Included:
- Source code (`.py` files)
- Configuration files (`requirements.txt`, `.env.example`, `Dockerfile`)
- Documentation (`README_GITHUB.md`, `F1_TRACKER_GUIDE.md`)
- Trained models (`models/*.pkl`)
- Dashboard files

### ❌ Excluded (.gitignore):
- Virtual environment (`venv/`, `.venv/`)
- Cache files (`cache/`, `__pycache__/`)
- Database files (`*.db`)
- Environment secrets (`.env`)
- Temporary files

---

## 🔄 Update Existing Repository

### After Making Changes:

```bash
# Stage changes
git add .

# Commit
git commit -m "Update: Description of changes"

# Push
git push origin main
```

### Quick Update Script:

```bash
git add . && git commit -m "Update app" && git push
```

---

## 🏷️ Create a Release

```bash
# Tag a version
git tag -a v1.0.0 -m "First production release"

# Push tags
git push origin --tags
```

---

## 🌐 Make Repository Public/Private

1. Go to repository on GitHub
2. Click **Settings**
3. Scroll to **Danger Zone**
4. Click **Change visibility**

---

## 📝 Update README for GitHub

After first push, rename README:

```bash
mv README_GITHUB.md README.md
git add README.md
git commit -m "Update README for GitHub"
git push
```

Don't forget to update:
- Replace `YOUR_USERNAME` with your actual GitHub username
- Add screenshot images
- Update repository URL links

---

## 🖼️ Add Screenshots

1. Run the app and take screenshots
2. Save to `screenshots/` folder:
   ```
   screenshots/
   ├── dashboard.png
   ├── predictions.png
   ├── analysis.png
   └── team_strength.png
   ```
3. Update README.md with actual image paths:
   ```markdown
   ![Dashboard](screenshots/dashboard.png)
   ```

---

## 🚀 GitHub Pages (Optional)

Host documentation:

1. Go to **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: **main** / **docs** folder
4. Save

Your docs will be at: `https://YOUR_USERNAME.github.io/f1predict/`

---

## 🔐 Environment Secrets

Never commit `.env` file with real secrets!

For deployment:
1. Use `.env.example` as template
2. Document required variables in README
3. For production, use platform-specific secret management:
   - Heroku: Config Vars
   - Railway: Environment Variables
   - Render: Environment

---

## 🐳 Deploy to Cloud

### Heroku
```bash
heroku create f1-tracker
git push heroku main
```

### Railway
```bash
railway init
railway up
```

### Render
1. Connect GitHub repository
2. Select `Dockerfile` as build method
3. Add environment variables
4. Deploy

---

## ✅ Deployment Checklist

- [ ] Remove sensitive data from code
- [ ] Update `.gitignore` properly
- [ ] Test app locally before pushing
- [ ] Update README with your repository URL
- [ ] Add screenshots
- [ ] Create LICENSE file
- [ ] Write clear commit messages
- [ ] Tag releases with version numbers
- [ ] Set up branch protection (optional)
- [ ] Add GitHub Actions for CI/CD (optional)

---

## 🆘 Troubleshooting

### Large files error
```bash
# If models are too large (>100MB)
git lfs track "*.pkl"
git add .gitattributes
git add models/*.pkl
git commit -m "Add LFS for large model files"
```

### Permission denied (SSH)
```bash
# Use HTTPS instead
git remote set-url origin https://github.com/YOUR_USERNAME/f1predict.git
```

### Authentication failed
```bash
# Use Personal Access Token instead of password
# Generate at: GitHub → Settings → Developer settings → Personal access tokens
```

---

## 📧 Need Help?

- [GitHub Docs](https://docs.github.com)
- [Git Tutorial](https://git-scm.com/docs/gittutorial)
- [GitHub Guides](https://guides.github.com)

---

**Ready to deploy? Run:**
```bash
./deploy_github.sh
```

🏁 **Good luck with your deployment!**
