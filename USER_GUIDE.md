# 📖 User Guide

## Getting Started

### First Time Setup

1. **Download** the Business Insights Agent folder
2. **Double-click** `start.bat` (Windows) or `start.sh` (Mac/Linux)
3. **Wait** for installation to complete (5-10 minutes)
4. **Open** your web browser to `http://localhost:8501`

### Using the Application

The application has 5 main tabs at the top:

## 📁 Data Load Tab

**Purpose**: Upload your data files

**What to do**:
1. Click "Choose a data file" button
2. Select your Excel (.xlsx, .xls) or CSV file
3. Wait for "✅ Data loaded successfully!" message
4. See a summary of your data

**Supported files**:
- Excel files (.xlsx, .xls, .xlsm)
- CSV files (.csv)
- OpenDocument files (.ods)

**Tips**:
- If you have multiple sheets in Excel, all will be loaded
- Try the sample data first to learn how it works
- Large files may take a few minutes to load

## 🔍 Data Explorer Tab

**Purpose**: Look at your data and choose what to analyze

**What to do**:
1. Select which table to explore (if you have multiple)
2. Choose which columns to analyze
3. See charts and statistics about your data
4. Look for patterns and insights

**Features**:
- **Data Preview**: See your data in a table
- **Column Selection**: Choose which columns to focus on
- **Data Types**: See what kind of data each column contains
- **Missing Values**: Check for empty or incomplete data
- **Statistics**: Get summary numbers for your data

## 🔗 Relationship Builder Tab

**Purpose**: Connect different data tables together

**What to do**:
1. Choose a source table and target table
2. Select which columns to connect
3. Choose the relationship type
4. Click "Add Relationship"
5. See a visual diagram of your connections

**Relationship Types**:
- **One-to-One**: Each record in one table matches one record in another
- **One-to-Many**: One record can match many records in another table
- **Many-to-One**: Many records can match one record in another table
- **Many-to-Many**: Records can have multiple matches

**Tips**:
- Connect tables with similar data (like customer IDs)
- The diagram helps you understand your data structure
- You can delete relationships by clicking the trash icon

## 🤖 AI Analysis Tab

**Purpose**: Ask questions and get intelligent answers about your data

**What to do**:
1. Make sure an AI model is selected and initialized
2. Type your question in the text box
3. Click "Generate AI Analysis"
4. Read the AI's response
5. Export your analysis as a report

**Example Questions**:
- "What are my top 5 products by sales?"
- "Show me customer trends over the last 6 months"
- "Which regions have the highest revenue?"
- "What patterns do you see in my data?"
- "Create a summary of my business performance"

**Features**:
- **Quick Insights**: Get instant analysis of your data
- **Analysis History**: See all your previous questions and answers
- **Export Reports**: Download your analysis as JSON or Markdown files
- **Multiple Formats**: Get insights in different formats

## ⚙️ Settings Tab

**Purpose**: Manage your AI models and export your data

**What to do**:
1. **AI Models**: See which AI models are available
2. **Export Data**: Download your data as CSV or Excel files
3. **System Info**: Check technical details about your setup

**Export Options**:
- **CSV**: Download data as a simple spreadsheet
- **Excel**: Download all tables as an Excel file with multiple sheets
- **Analysis Reports**: Download your AI analysis as reports

## 💡 Tips for Better Results

### Data Preparation
- **Clean your data** before uploading (remove empty rows, fix typos)
- **Use clear column names** (like "Customer Name" instead of "Cust1")
- **Make sure dates are in a consistent format**
- **Check for missing data** and fill it in if possible

### Asking AI Questions
- **Be specific**: "Top 10 products by revenue" is better than "best products"
- **Use business terms**: "sales", "revenue", "customers", "profit"
- **Ask follow-up questions**: "Now show me the trends for those products"
- **Try different angles**: "What's the worst performing region?"

### Working with Multiple Tables
- **Connect related tables** using common columns (like customer ID)
- **Start simple** with one table, then add relationships
- **Use the ER diagram** to understand your data structure
- **Test relationships** by asking questions that span multiple tables

## 🔧 Troubleshooting

### Common Issues

**"No data loaded"**
- Go to Data Load tab and upload a file
- Try the sample data first

**"AI model not initialized"**
- Go to AI Analysis tab
- Select a model and click "Initialize Model"
- Wait for the success message

**"Can't connect to AI"**
- Make sure Ollama is running: `ollama serve`
- Restart the application

**"File won't upload"**
- Check file format (Excel or CSV)
- Try a smaller file first
- Make sure the file isn't corrupted

**"App is slow"**
- Close other programs to free up memory
- Try with smaller data files
- Restart the application

### Getting Help

1. **Check the terminal/command prompt** for error messages
2. **Try the sample data** to see if the app works
3. **Restart everything**: Close the app, run `ollama serve`, then restart
4. **Check your data**: Make sure it's in the right format

## 🎉 You're Ready!

You now know how to use the Business Insights Agent. Start with the sample data, then upload your own files and start asking questions about your business!

**Happy analyzing!** 📊✨
