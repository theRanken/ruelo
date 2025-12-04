#!/bin/bash

echo "🔧 Installing Python dependencies from requirements.txt..."

# Activate the virtual environment
source env/Scripts/activate

if [ -f requirements.txt ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Python dependencies installed successfully."
else
    echo "❌ requirements.txt not found!"
    exit 1
fi
