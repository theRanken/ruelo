#!/bin/bash

echo "🔧 Installing Python dependencies from requirements.txt..."

# Optionally activate a virtual environment here if needed
# source venv/bin/activate

if [ -f requirements.txt ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Python dependencies installed successfully."
else
    echo "❌ requirements.txt not found!"
    exit 1
fi
