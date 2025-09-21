# 🚀 Quick Start Guide

## For Everyone (Easiest Method)

### Step 1: Download
- Download the Business Insights Agent folder to your computer
- Make sure you have internet connection for the first setup

### Step 2: Start the Application
**Windows Users:**
- Double-click `start.bat`
- Wait for everything to install (5-10 minutes)
- Your web browser will open automatically

**Mac/Linux Users:**
- Double-click `start.sh` 
- Wait for everything to install (5-10 minutes)
- Your web browser will open automatically

### Step 3: Start Analyzing
- You'll see the application in your web browser
- Click "Data Load" tab
- Try the sample data first
- Then upload your own Excel or CSV files

## For Technical Users

### Prerequisites
- Python 3.10 or higher
- Internet connection (first time only)

### Installation
```bash
# Clone or download the repository
git clone <repository-url>
cd business-insights-agent

# Run the automated setup
python start.py
```

### Usage
```bash
# Start the application
python start.py

# Open your browser to:
http://localhost:8501
```

## What Happens During Setup

1. **Checks Python version** - Makes sure you have the right version
2. **Installs dependencies** - Downloads required software packages
3. **Installs Ollama** - Sets up the AI engine (Mac/Linux only)
4. **Downloads AI models** - Gets the AI models for analysis
5. **Starts the application** - Opens the web interface

## First Time Using the App

### 1. Try Sample Data
- Click "Data Load" tab
- Click "Load Sample Data" button
- This gives you data to practice with

### 2. Explore the Interface
- Click through all 5 tabs to see what's available
- Don't worry about breaking anything - it's safe to explore

### 3. Upload Your Data
- Click "Choose a data file" button
- Select your Excel or CSV file
- Wait for it to load

### 4. Ask Questions
- Go to "AI Analysis" tab
- Type a question about your data
- Click "Generate AI Analysis"
- See what insights you get!

## Common First Questions to Try

- "What are the top 5 products by sales?"
- "Show me trends over time"
- "Which customers spend the most?"
- "What patterns do you see in this data?"
- "Create a summary of my business performance"

## Troubleshooting

### If the app won't start:
1. Make sure Python is installed
2. Try running: `pip install -r requirements.txt`
3. Check the terminal for error messages

### If AI analysis doesn't work:
1. Make sure Ollama is running: `ollama serve`
2. Restart the application
3. Try a simpler question first

### If data won't load:
1. Check that your file is Excel (.xlsx) or CSV (.csv)
2. Try the sample data first
3. Make sure the file isn't too large

## Next Steps

Once you're comfortable with the basics:

1. **Read the User Guide** - Learn advanced features
2. **Try different data files** - Upload your real business data
3. **Connect multiple tables** - Use the Relationship Builder
4. **Export your analysis** - Download reports and insights
5. **Ask complex questions** - Get deeper insights from your data

## Need Help?

- Check the terminal/command prompt for error messages
- Try the sample data to test if everything works
- Restart the application if something seems stuck
- Read the full User Guide for detailed instructions

## 🎉 You're Ready!

The Business Insights Agent is now running on your computer. Start exploring your data and discovering insights!

**Happy analyzing!** 📊✨