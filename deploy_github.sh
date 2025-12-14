#!/bin/bash
# Git Deployment Script for F1 Prediction Tracker

echo "🏎️ F1 Prediction Tracker - GitHub Deployment"
echo "=============================================="
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
    echo "✅ Git initialized"
    echo ""
fi

# Add all files
echo "📝 Staging files..."
git add .

# Show status
echo ""
echo "📊 Current status:"
git status --short

# Commit
echo ""
read -p "📝 Enter commit message (or press Enter for default): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Initial commit: F1 Prediction Tracker with Streamlit dashboard"
fi

git commit -m "$commit_msg"
echo "✅ Files committed"
echo ""

# Check if remote exists
if ! git remote | grep -q origin; then
    echo "🌐 No remote repository configured."
    echo ""
    echo "Please create a new repository on GitHub, then enter the URL below."
    echo "Example: https://github.com/YOUR_USERNAME/f1predict.git"
    echo ""
    read -p "Enter GitHub repository URL: " repo_url
    
    if [ -n "$repo_url" ]; then
        git remote add origin "$repo_url"
        echo "✅ Remote repository added: $repo_url"
    else
        echo "❌ No URL provided. Skipping remote setup."
        echo "You can add it later with: git remote add origin <URL>"
        exit 0
    fi
fi

# Get current branch name
branch=$(git branch --show-current)

# Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."
read -p "Push to branch '$branch'? (y/n): " confirm

if [ "$confirm" = "y" ]; then
    git push -u origin "$branch"
    echo ""
    echo "✅ Successfully deployed to GitHub!"
    echo ""
    echo "🔗 Your repository is now live at:"
    repo_url=$(git remote get-url origin)
    repo_url_web=$(echo "$repo_url" | sed 's/\.git$//')
    echo "$repo_url_web"
    echo ""
    echo "📖 Next steps:"
    echo "1. Update README_GITHUB.md with your repository URL"
    echo "2. Add screenshots to make it more attractive"
    echo "3. Enable GitHub Pages for documentation (optional)"
    echo "4. Set up GitHub Actions for CI/CD (optional)"
else
    echo "❌ Push cancelled"
fi

echo ""
echo "🏁 Deployment script complete!"
