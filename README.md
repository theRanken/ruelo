<div align="center">
    <img src="src/assets/logo.jpg" alt="" width="200">
</div>

A powerful face recognition and analysis library for PHP using DeepFace, with support for file paths, base64 strings, and data URLs.

## What is this?

This package provides robust face recognition, verification, and analysis capabilities using DeepFace and deep learning models. It supports multiple input formats and provides comprehensive error handling and validation. The backend is powered by a persistent Python FastAPI server for high performance and reliability.

## Features

- **Face Verification:** Compare faces between two images with confidence scores
- **Face Analysis:** Get age, gender, emotion, and race predictions
- **Multiple Input Formats:** Support for file paths, base64 strings, and data URLs
- **Base64 Utilities:** Built-in methods for converting between formats
- **Input Validation:** Comprehensive error checking and validation
- **Detailed Results:** Get match status, confidence scores, and similarity metrics
- **Error Handling:** Clear error messages and consistent error format
- **FastAPI Backend:** Persistent Python API server for high performance

## Requirements

### PHP Requirements

- PHP 7.2 or higher
- php-fileinfo extension
- php-json extension
- Composer (for PHP dependencies)

### Python Requirements

- Python >=3.8 or <=3.11
- tensorflow==2.11
- deepface==0.0.96
- opencv-python-headless==4.10.0.84
- numpy==1.24.4
- pillow==10.4.0
- pyyaml==6.0.2
- fastapi==0.115.2
- uvicorn==0.30.3
- python-multipart==0.0.9
- onnxruntime==1.17.0
- pydantic==2.10.6
- (see requirements.txt for details)

## Installation

1. **Clone or download the repository:**

    ```bash
    git clone https://github.com/theranken/ruelo.git
    cd ruelo
    ```

2. **Install the PHP library via Composer:**

    ```bash
    composer install
    ```

    Or if using as a dependency:

    ```bash
    composer require theranken/ruelo
    ```

3. **Install Python dependencies:**

    You can use the provided setup script:

    ```bash
    ./setup.sh  # For Unix/Linux/macOS
    # or
    pip install -r requirements.txt
    ```

    On Windows, run:

    ```bash
    pip install -r requirements.txt
    ```

4. **Start the FastAPI server:**

    The server can start automatically when you use the library methods, or you can start it manually.

    **Automatic (recommended):** The server starts automatically when you call `DeepFace::compare()` or `DeepFace::analyze()`.

    **Manual start:** Use the provided utility scripts:

    ```bash
    php utilities/start_server.php
    ```

    If installed via Composer in another application:

    ```bash
    php vendor/theranken/ruelo/utilities/start_server.php
    ```

    To stop the server manually:

    ```bash
    php utilities/stop_server.php
    ```

    Or if installed via Composer:

    ```bash
    php vendor/theranken/ruelo/utilities/stop_server.php
    ```

    Or directly:

    ```bash
    python src/scripts/df_service.py
    ```

5. **(Optional) Test with the interactive CLI tool:**

    Run the interactive shell to test the library:

    ```bash
    php interact
    ```

6. **(Optional) Docker usage:**
    See below for Docker instructions.

## Usage

### Basic Face Comparison

```php
use Ruelo\DeepFace;

$deepface = new DeepFace('http://localhost:4800'); // URL of FastAPI server

// Compare two image files
$result = $deepface->compare('path/to/image1.jpg', 'path/to/image2.jpg');
if (isset($result['result']['verified']) && $result['result']['verified']) {
    echo "Match found! Distance: " . $result['result']['distance'] . "\n";
    echo "Threshold: " . $result['result']['threshold'] . "\n";
    echo "Model: " . $result['result']['model'] . "\n";
}

// Using a custom threshold (0.0 to 1.0)
$result = $deepface->compare('image1.jpg', 'image2.jpg', 0.5);
```

### Working with Base64 Images

```php
use Ruelo\DeepFace;

$deepface = new DeepFace('http://localhost:4800');

// Convert an image file to base64
$base64 = $deepface->fileToBase64('path/to/image.jpg');

// Compare with mixed formats
$result = $deepface->compare($base64, 'path/to/image2.jpg');

// Compare two base64 images
$base64_1 = $deepface->fileToBase64('image1.jpg');
$base64_2 = $deepface->fileToBase64('image2.jpg');
$result = $deepface->compare($base64_1, $base64_2);

// Working with data URLs
$dataUrl = 'data:image/jpeg;base64,/9j/4AAQSkZJRg...';
$result = $deepface->compare($dataUrl, 'image2.jpg');
```

### Face Analysis

```php
use Ruelo\DeepFace;

// Basic analysis (static method)
$result = DeepFace::analyze('path/to/image.jpg');

// The result contains age, gender, emotion, and race predictions
echo "Age: " . $result['age'] . "\n";
echo "Gender: " . $result['gender'] . "\n";
echo "Emotion: " . $result['dominant_emotion'] . "\n";
```

// The FastAPI server URL can be customized if needed:
$deepface = new DeepFace('<http://localhost:4800>');

### Using Static Helper Methods

```php
use Ruelo\DeepFace;

// Quick face comparison
$result = DeepFace::compareImages('image1.jpg', 'image2.jpg', 'http://localhost:8000');
```

// The CLI tool is no longer required. All operations are handled via the FastAPI server.

## Interactive CLI Tool

The library includes an interactive command-line tool for testing and experimenting with face recognition and analysis features. This tool provides a simple interface to test the library without writing code.

### Running the Interactive Tool

```bash
php interact
```

### Example Session

```
=== DeepFacePHP Interactive Shell ===
Type 'exit' to quit.

Choose action ([v]erify, [a]nalyze, [q]uit): v
Enter path to first image: /path/to/image1.jpg
Enter path to second image: /path/to/image2.jpg
Array
(
    [result] => Array
        (
            [verified] => 1
            [distance] => 0.234
            [threshold] => 0.6
            [model] => ArcFace
            ...
        )
    [total_time_seconds] => 1.234
)

Choose action ([v]erify, [a]nalyze, [q]uit): a
Enter path to image: /path/to/image.jpg
Array
(
    [age] => 28
    [gender] => Woman
    [dominant_emotion] => happy
    ...
)

Choose action ([v]erify, [a]nalyze, [q]uit): q
Goodbye!
```

The interactive tool supports:
- **Face verification** between two images
- **Face analysis** for age, gender, emotion, and race prediction
- **Base64 testing** with pre-encoded image files
- **Clear screen** and **quit** commands

## Results Format

### Face Comparison Results

```php
[
    'result' => [
        'verified' => true|false,        // Whether the faces match (1 or 0)
        'distance' => 0.0,               // Distance between faces (lower is more similar)
        'threshold' => 0.3,              // Threshold used for verification
        'model' => 'Facenet512',         // Model used for comparison
        'detector_backend' => 'opencv',  // Face detection backend
        'similarity_metric' => 'cosine', // Similarity metric used
        'facial_areas' => [              // Detected facial areas
            'img1' => [
                'x' => 269,
                'y' => 163,
                'w' => 193,
                'h' => 193,
                'left_eye' => null,
                'right_eye' => null
            ],
            'img2' => [
                'x' => 269,
                'y' => 163,
                'w' => 193,
                'h' => 193,
                'left_eye' => null,
                'right_eye' => null
            ]
        ],
        'time' => 2.88                  // Processing time in seconds
    ],
    'total_time_seconds' => 2.9133     // Total time including overhead
]
```

### Analysis Results

```php
[
    'age' => 25,
    'gender' => 'Man',
    'dominant_emotion' => 'happy',
    'emotion' => [
        'angry' => 0.01,
        'disgust' => 0.0,
        'fear' => 0.01,
        'happy' => 0.95,
        'sad' => 0.02,
        'surprise' => 0.01,
        'neutral' => 0.0
    ]
]
```

### Error Format

```php
[
    'error' => 'Error message description'
]
```

Common error messages:

- 'Both image sources are required'
- 'Image file not found'
- 'Failed to execute Python script'
- 'Invalid JSON response'
- 'Database path not found'

## How It Works

1. The PHP library validates inputs and handles format conversions
2. A Python FastAPI server using DeepFace processes the images
3. Results are returned as JSON and parsed into PHP arrays
4. Comprehensive error handling ensures reliable operation

## Troubleshooting

### Installation Issues

- Ensure Python 3.8+ is installed and accessible
- Install all required Python packages: `pip install -r requirements.txt`
- Check file permissions for the Python script
- On Windows, ensure the Python executable is in your PATH
- On Unix/Linux/macOS, make sure the setup.sh script is executable: `chmod +x setup.sh`

### Input Problems

- Verify image files exist and are readable
- Ensure base64 strings are properly formatted
- Check that data URLs include the correct MIME type

### Docker Usage

You can run the FastAPI server in a Docker container for production use. Example Dockerfile:

```Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["python", "src/scripts/Python/deepface_api_service.py"]
```

Build and run:

```bash
docker build -t deepface-api .
docker run -p 8000:8000 deepface-api
```

## License

MIT

---

## Built with ❤️ using DeepFace, FastAPI, and PHP**
