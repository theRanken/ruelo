@echo off
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Downloading ONNX models...
python src/scripts/Python/download_models.py

echo.
echo Setup complete!
