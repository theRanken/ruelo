# DeepFace PHP Library

Easily use powerful face recognition and analysis in your PHP projects by connecting to the DeepFace Python library—no need to learn Python!

## What is this?
This package lets you verify if two faces match or analyze faces in images using the popular [DeepFace](https://github.com/serengil/deepface) library, all from PHP. It works by running a Python script behind the scenes and giving you the results in PHP.

## Features
- **Face Verification:** Check if two images are of the same person.
- **Face Analysis:** Detect age, gender, emotion, and more from a single image.

## Requirements
- PHP 7.2 or higher
- Python 3 with the `deepface` library installed
- Composer (for PHP dependencies)

## Installation
1. **Install the PHP library:**
   ```bash
   composer require theranken/deepface-php
   ```
2. **Install Python dependencies:**
   Make sure you have Python 3 and pip installed, then run:
   ```bash
   pip install deepface
   ```

## Usage

### 1. Prepare your images
Make sure your images are accessible by file path (local files).

### 2. Example: Face Verification (Instance Method)
```php
use DeepFacePHP\DeepFace;

$deepface = new DeepFace();
$result = $deepface->verify('path/to/image1.jpg', 'path/to/image2.jpg');

if ($result['verified']) {
    echo "Faces match!";
} else {
    echo "Faces do not match.";
}
```

### 3. Example: Face Verification (Static Method)
```php
use DeepFacePHP\DeepFace;

$result = DeepFace::verifyImage'path/to/image1.jpg', 'path/to/image2.jpg');

if ($result['verified']) {
    echo "Faces match!";
} else {
    echo "Faces do not match.";
}
```

### 4. Example: Face Analysis (Instance Method)
```php
use DeepFacePHP\DeepFace;

$deepface = new DeepFace();
$result = $deepface->analyze('path/to/image.jpg', ['age', 'gender', 'emotion']);

print_r($result);
```

### 5. Example: Face Analysis (Static Method)
```php
use DeepFacePHP\DeepFace;

$result = DeepFace::analyzeImage('path/to/image.jpg', ['age', 'gender', 'emotion']);

print_r($result);
```

### 6. Customizing Python Path
If your Python or script path is different, you can specify them:
```php
$deepface = new DeepFace('python3', '/custom/path/deepface_cli.py');
// Or for static methods:
$result = DeepFace::verifyImage('img1.jpg', 'img2.jpg', 'python3', '/custom/path/deepface_cli.py');
```

## How does it work?
- The PHP library runs a Python script using shell commands.
- The Python script uses DeepFace to process the images and returns the results as JSON.
- The PHP library reads and parses the results for you.

## Troubleshooting
- Make sure Python and DeepFace are installed and accessible from your command line.
- If you get errors, check the output for details (e.g., missing dependencies, wrong image paths).

## License
MIT

---

**Enjoy using DeepFace in your PHP projects!**
