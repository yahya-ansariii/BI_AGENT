# 📊 BI Agent

**Your personal, private, and powerful AI Business Intelligence Agent.**

BI Agent is a comprehensive, offline-first business intelligence tool that helps you analyze data, generate insights, and create visualizations using local AI models. All processing happens on your computer, ensuring complete data privacy and security.

## ✨ Key Features

- **🔒 100% Offline Operation** - Your data never leaves your computer
- **🤖 AI-Powered Analysis** - Local AI models for intelligent data insights
- **📊 Advanced Visualizations** - Interactive charts and graphs with auto-generation
- **🔗 Smart Relationship Builder** - AI-powered relationship detection with manual editing
- **📄 Professional PDF Reports** - Beautiful, formatted documents with enhanced styling
- **🛠️ Custom SQL** - Write and execute custom SQL queries with auto-visualization
- **📁 Multi-Format Support** - Excel, CSV, and custom table creation
- **🎨 Modern UI** - Clean, intuitive interface with tab navigation
- **🔍 Search & Filter** - Find relationships and insights quickly
- **📤 Bulk Export** - Export all insights or individual reports
- **⚙️ Comprehensive Settings** - Configure AI models and data connectors

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or newer
- 4GB+ RAM recommended
- 2GB+ free disk space for AI models

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/bi-agent.git
   cd bi-agent
   ```

2. **Run the setup:**
   ```bash
   # Windows
   start.bat
   
   # macOS/Linux
   ./start.sh
   ```

3. **Access the application:**
   Open your browser to `http://localhost:8501`

The setup script will automatically:
- Install all required dependencies
- Download and configure Ollama (AI model server)
- Download essential AI models
- Start the application

## 📚 Usage Guide

### 1. Data Loading
- **Upload Files**: Support for Excel (.xlsx, .xls) and CSV files
- **Multi-Sheet Excel**: Automatically loads all sheets as separate tables
- **Custom Tables**: Create tables with custom schemas
- **Sample Data**: Try the application with pre-built demo data

### 2. Data Exploration
- **Table Management**: Rename, delete, and organize your tables
- **Data Preview**: View and filter your data
- **Statistics**: Comprehensive data summaries and quality metrics
- **Visualizations**: Automatic chart generation

### 3. Smart Relationship Building
- **AI Detection**: Automatically detect relationships using AI analysis
- **Manual Editing**: Edit, delete, and modify detected relationships
- **Search & Filter**: Find specific relationships quickly
- **Bulk Operations**: Export, import, and manage relationships in bulk
- **ER Diagrams**: Beautiful visual representation of data relationships
- **Visual Indicators**: Distinguish between AI-detected and manual relationships

### 4. AI Analysis
- **Natural Language Queries**: Ask questions in plain English
- **SQL Generation**: AI converts questions to SQL queries
- **Custom SQL**: Write and execute your own SQL queries with auto-visualization
- **Enhanced Insights**: AI analyzes both query results and source data
- **Query History**: Track and re-run previous queries
- **Real-time Visualization**: Automatic chart generation for query results

### 5. Professional Report Generation
- **Individual PDF Export**: Export current analysis with enhanced styling
- **Bulk PDF Export**: Export all insights in a comprehensive document
- **Professional Styling**: Beautiful typography, colors, and layout
- **Smart Formatting**: Automatic parsing of headings, lists, and code blocks
- **Metadata Tables**: Professional information display
- **Word Documents**: Microsoft Word compatible exports
- **Excel Reports**: Structured data with multiple sheets

## 🆕 Recent Improvements

### Enhanced User Experience
- **Professional PDF Styling**: Beautiful reports with improved typography and colors
- **Smart Relationship Detection**: AI automatically detects table relationships
- **Advanced Relationship Management**: Edit, search, filter, and bulk operations
- **Improved Navigation**: Better tab switching and user guidance
- **Enhanced Error Handling**: Comprehensive error messages and recovery

### New Features
- **Auto-Visualization**: Automatic chart generation for SQL query results
- **Query History**: Track and re-run previous queries
- **Bulk Export**: Export all insights in a single comprehensive PDF
- **Search & Filter**: Find relationships and insights quickly
- **Settings Management**: Comprehensive configuration options
- **Session State Management**: Improved data persistence and state handling

### Technical Improvements
- **Robust JSON Parsing**: Better handling of AI responses
- **Enhanced Error Recovery**: Graceful handling of edge cases
- **Improved Code Organization**: Better structure and maintainability
- **Performance Optimizations**: Faster processing and better responsiveness

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit web interface with enhanced UI components
- **Data Processing**: Pandas + DuckDB for SQL queries
- **AI Engine**: Ollama with local LLM models and robust parsing
- **Visualization**: Plotly + Matplotlib + Seaborn with auto-generation
- **Export**: ReportLab (PDF) + python-docx (Word) with professional styling

### Supported AI Models
- Llama 3.2 (3B, 8B)
- CodeLlama (7B, 13B)
- DeepSeek R1 (8B)
- Any Ollama-compatible model

### Data Formats
- **Input**: Excel (.xlsx, .xls, .xlsm, .xlsb), CSV
- **Output**: PDF, Word (.docx), Excel (.xlsx), JSON

## 🔧 Configuration

### AI Model Management
```bash
# Download additional models
ollama pull llama3.2:8b
ollama pull codellama:13b

# List available models
ollama list
```

### Custom Configuration
Edit `config/model_settings.json` to customize:
- Default AI model
- Query timeouts
- Analysis parameters

## 📁 Project Structure

```
bi-agent/
├── app.py                 # Main Streamlit application
├── start.py              # Setup and launch script
├── requirements.txt      # Python dependencies
├── modules/              # Core functionality
│   ├── data_processor.py # Data loading and processing
│   ├── llm_agent.py     # AI analysis engine
│   ├── visualizer.py    # Chart generation
│   └── schema_manager.py # Data relationships
├── config/               # Configuration files
│   └── ollama_config.py  # Ollama settings
└── connectors/           # External data connectors
    ├── excel_connector.py
    ├── ga_connector.py
    └── snowflake_connector.py
```

## 🔒 Privacy & Security

- **Local Processing**: All data analysis happens on your machine
- **No Cloud Dependencies**: Works completely offline after setup
- **No Data Transmission**: Your data never leaves your computer
- **Open Source**: Full source code available for review

## 🐛 Troubleshooting

### Common Issues

**"Ollama not found"**
```bash
# Install Ollama manually
curl -fsSL https://ollama.ai/install.sh | sh
```

**"No models available"**
```bash
# Download a model
ollama pull llama3.2:3b
```

**"Port 8501 already in use"**
```bash
# Kill existing Streamlit processes
pkill -f streamlit
```

### Getting Help
- Check the [Issues](https://github.com/yourusername/bi-agent/issues) page
- Review the [User Guide](USER_GUIDE.md)
- Run the verification script: `python verify_offline.py`

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/yourusername/bi-agent.git
cd bi-agent
pip install -r requirements.txt
python -m streamlit run app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local AI model serving
- [Streamlit](https://streamlit.io/) for the web interface
- [DuckDB](https://duckdb.org/) for fast SQL processing
- [Plotly](https://plotly.com/) for interactive visualizations

## 📊 Screenshots

![Data Loading](screenshots/data-loading.png)
![AI Analysis](screenshots/ai-analysis.png)
![ER Diagram](screenshots/er-diagram.png)
![Report Export](screenshots/report-export.png)

---

**Made with ❤️ for data-driven decision making**