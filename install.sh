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
echo "Installing with requirements.txt..."
pip install -r requirements.txt

echo "✅ Installation complete."


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
echo "3. Start the app: python app.py"
echo ""
echo "📚 For more information, see the readme.md file"

