"""
Setup script for Business Insights Agent
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="business-insights-agent",
    version="1.0.0",
    author="Business Insights Agent Team",
    author_email="contact@businessinsights.com",
    description="Offline AI-powered business intelligence tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/business-insights-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Business",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "business-insights=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.md"],
    },
    keywords=[
        "business intelligence",
        "data analysis",
        "ai",
        "machine learning",
        "streamlit",
        "duckdb",
        "ollama",
        "langchain",
        "visualization",
        "offline",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/business-insights-agent/issues",
        "Source": "https://github.com/yourusername/business-insights-agent",
        "Documentation": "https://business-insights-agent.readthedocs.io/",
    },
)
