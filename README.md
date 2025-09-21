# Business Insights Agent

A powerful offline business intelligence tool that combines data analysis, visualization, and AI-powered insights using Python, Streamlit, DuckDB, and local LLM capabilities.

## Features

- 📊 **Data Processing**: Excel file support with DuckDB for fast SQL queries
- 📈 **Visualization**: Interactive charts with Plotly and Matplotlib
- 🤖 **AI Insights**: Local LLM integration with Ollama
- 🗂️ **Schema Management**: JSON-based data schema and relationship storage
- 💻 **Offline Operation**: Complete offline functionality with local data processing
- 🔗 **Multiple Connectors**: Excel, Snowflake, Google Analytics, Oncore support

## Tech Stack

- **Python 3.10+**
- **Streamlit** - Web interface
- **Pandas + DuckDB** - Data processing and querying
- **Matplotlib/Plotly** - Data visualization
- **Ollama** - Local LLM integration
- **JSON** - Schema and relationship storage

## Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd business-insights-agent
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup Ollama (for AI features)**
   ```bash
   # Install Ollama from https://ollama.ai/
   # Pull a model (e.g., llama3:8b-instruct)
   ollama pull llama3:8b-instruct
   ```

4. **Create demo data**
   ```bash
   python create_demo_data.py
   ```

5. **Run the application**
   ```bash
   streamlit run app/frontend.py
   ```

## Usage

### 1. Schema Approval
- Load demo schema or upload your own Excel files
- Review table structures and column information
- Define relationships between tables (e.g., sales.date = web_traffic.date)
- Save schema and relationships for future use

### 2. Ask Questions
- Enter natural language questions about your data
- Examples:
  - "Total sales by region"
  - "Compare sales and visits by date"
  - "Average amount by region"
  - "Show top performing sources"
- Get automatically generated SQL queries
- View query results with insights and visualizations

### 3. AI-Powered Analysis
- Natural language to SQL conversion
- Automated insights generation
- Business recommendations based on data patterns
- Data quality assessment

### 4. Visualization
- Automatic chart generation based on data types
- Bar charts for categorical data
- Line charts for time series data
- Scatter plots for correlation analysis

### 5. Multiple Data Sources
- Excel files (.xlsx, .xls)
- Snowflake databases
- Google Analytics
- Oncore systems

## Configuration

### LLM Models
The application supports various Ollama models:
- `llama2` - General purpose model
- `codellama` - Code-focused model
- `mistral` - Efficient model
- `phi` - Microsoft's Phi model

### Data Sources
- Excel files (.xlsx, .xls)
- Sample business data (automatically generated)
- Future: Database connections, CSV files

## Project Structure

```
business-insights-agent/
├── app/
│   ├── __init__.py
│   └── frontend.py        # Main Streamlit application
├── agent/
│   ├── __init__.py
│   ├── llm_agent.py       # LLM integration with Ollama
│   ├── sql_generator.py   # SQL generation from natural language
│   └── insights.py        # Insights generation
├── connectors/
│   ├── __init__.py
│   ├── excel_connector.py # Excel file operations
│   ├── snowflake_connector.py # Snowflake database
│   ├── ga_connector.py    # Google Analytics
│   └── oncore_connector.py # Oncore system
├── schema_store/          # JSON schema storage
│   ├── schema.json
│   └── relationships.json
├── demo_data/            # Demo Excel files
│   ├── sales.xlsx
│   └── web_traffic.xlsx
├── tests/                # Test cases
│   ├── __init__.py
│   └── test_agent.py
├── requirements.txt      # Python dependencies
├── create_demo_data.py   # Demo data creation script
└── README.md            # This file
```

## API Reference

### DataProcessor
- `load_excel_data(file_upload)` - Load Excel data
- `load_sample_data()` - Load sample business data
- `execute_sql_query(query)` - Execute SQL on data
- `get_data_summary()` - Get comprehensive data summary

### Visualizer
- `create_sales_trend_chart(data)` - Sales trend visualization
- `create_category_analysis(data)` - Category breakdown
- `create_correlation_heatmap(data)` - Correlation analysis
- `create_dashboard_summary(data)` - Complete dashboard

### LLMAgent
- `analyze_data(query, data)` - AI-powered data analysis
- `sales_analysis(data)` - Automated sales insights
- `customer_analysis(data)` - Customer segmentation analysis
- `get_quick_insights(data)` - Quick data overview

### SchemaManager
- `update_schema_from_data(data)` - Update schema from data
- `add_relationship(from_table, to_table)` - Add data relationships
- `detect_relationships(data)` - Auto-detect relationships
- `export_schema(filepath)` - Export schema configuration

## Troubleshooting

### Common Issues

1. **Ollama not running**
   - Ensure Ollama is installed and running
   - Check if the model is pulled: `ollama list`
   - Start Ollama service: `ollama serve`

2. **Memory issues with large datasets**
   - Use data sampling for initial analysis
   - Consider chunking large Excel files
   - Monitor memory usage in the dashboard

3. **Visualization errors**
   - Check data types and column names
   - Ensure numeric columns for charts
   - Verify date columns are properly formatted

### Performance Tips

- Use DuckDB for large dataset queries
- Sample data for initial exploration
- Cache frequently used visualizations
- Optimize LLM prompts for better performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Testing

Run the test suite to verify functionality:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_agent.py

# Run with verbose output
pytest -v tests/
```

The test suite includes:
- SQL generation validation
- Schema loading tests
- Relationship handling tests
- Excel connector functionality

## Example Queries

Try these example queries in the "Ask Questions" tab:

- **"Total sales by region"** - Groups sales data by region
- **"Compare sales and visits by date"** - Joins sales and web traffic data
- **"Average amount by region"** - Calculates average sales by region
- **"Show top performing sources"** - Ranks traffic sources by visits
- **"Sales trend over time"** - Shows sales progression over dates

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on GitHub

## Roadmap

- [ ] Database connectivity (PostgreSQL, MySQL)
- [ ] Advanced ML models integration
- [ ] Real-time data streaming
- [ ] Collaborative features
- [ ] Mobile app support
- [ ] Cloud deployment options
