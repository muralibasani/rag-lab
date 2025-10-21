#!/bin/bash

# Installation Script

set -e  # Exit on any error

echo "🧠 AI Assistant - Setup"
echo "============================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
echo "Choose installation method:"

echo "1) Traditional: pip install -r requirements.txt"
echo "2) Development: pip install -e .[dev]"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Installing with requirements.txt..."
        pip install -r requirements.txt
        ;;
    2)
        echo "Installing with development dependencies..."
        pip install -e .[dev]
        ;;
    *)
        echo "Invalid choice. Using pyproject.toml..."
        pip install -e .
        ;;
esac

# Install Playwright browsers
echo "🌐 Installing Playwright browsers..."
playwright install chromium

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF

# LLM Assistant Configuration

# Chunking Configuration
CHUNK_SIZE=2000
CHUNK_OVERLAP=400
...

# User Agent for Web Scraping
USER_AGENT=KafkaLLMAssistant/1.0 (+https://localhost)

# Question Prefix and Suffix
QUESTION_PREFIX=In topic context, 
QUESTION_SUFFIX=Also consider related terms or synonyms that might appear in the documents.

# AI Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
LLM_MODEL=llama3

# Database Configuration
DATABASE_URL=sqlite:///./db/queries.db

EOF
    echo "✅ .env file created with default settings"
else
    echo "✅ .env file already exists"
fi

# Create resources directory and files
echo "📁 Creating resources directory..."
mkdir -p resources


echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file to configure your settings"
echo "2. Add assistants.json in resources/ directory"
echo "3. Start the backend: python app.py"
echo "4. Or run with uvicorn: uvicorn app:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "🔗 Useful URLs:"
echo "   - API Documentation: http://localhost:8000/docs"
echo "   - Admin Interface: http://localhost:8000/admin (TBD)"
echo "   - Health Check: http://localhost:8000/"
echo ""
echo "📚 For more information, see the README.md file"

