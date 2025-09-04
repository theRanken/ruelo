<?php
namespace Ruelo;

class DeepFace
{

    private $apiUrl;
    private static $defaultApiUrl = 'http://127.0.0.1:4800';
    private static $serverStarted = false;

    public function __construct($apiUrl = null)
    {
        $this->apiUrl = rtrim($apiUrl ?? self::$defaultApiUrl, '/');
    }

    private static function startServerIfNeeded($apiUrl = null)
    {
        $url       = $apiUrl ?? self::$defaultApiUrl;
        $healthUrl = rtrim($url, '/') . '/health';
        DeepFace::log("Checking FastAPI health at $healthUrl");
        $health = @file_get_contents($healthUrl);
        if ($health === false) {
            $pythonScript = __DIR__ . '/scripts/Python/df_service.py';
            DeepFace::log("FastAPI not running. Attempting to start: $pythonScript");
            $cmd = "python \"$pythonScript\" > /dev/null 2>&1 &";
            exec($cmd, $output, $ret);
            DeepFace::log("Executed command: $cmd | Return: $ret | Output: " . implode(' ', $output));

            $start = time();
            while (true) {
                usleep(500000); // 0.5s
                $health = @file_get_contents($healthUrl);
                DeepFace::log("Waiting for FastAPI health... Status: " . ($health !== false ? 'OK' : 'NOT OK'));
                if ($health !== false || (time() - $start) > 15) {
                    break;
                }
            }
            if ($health === false) {
                DeepFace::log("FastAPI server failed to start or is unreachable after 15 seconds.");
            } else {
                DeepFace::log("FastAPI server is up and healthy.");
            }
        } else {
            DeepFace::log("FastAPI server is already running.");
        }
    }

    public static function log($message)
    {
        $logDir = __DIR__ . '/assets/temp';
        if (! is_dir($logDir)) {
            mkdir($logDir, 0755, true);
        }
        $logFile   = $logDir . '/deepface.log';
        $timestamp = date('Y-m-d H:i:s');
        file_put_contents($logFile, "[$timestamp] $message\n", FILE_APPEND);
    }

    private function isFilePath($input)
    {
        if (strpos($input, 'data:image/') === 0) {
            return false;
        }
        if (strpos($input, '/') !== false || strpos($input, '\\') !== false) {
            return true;
        }
        if (preg_match('/\.(jpg|jpeg|png|gif|bmp|webp)$/i', $input)) {
            return true;
        }
        if (strlen($input) > 1000 && preg_match('/^[A-Za-z0-9+\/]+=*$/', $input)) {
            return false;
        }
        return true;
    }

    public function fileToBase64($filePath)
    {
        if (! file_exists($filePath)) {
            return false;
        }
        $imageData = file_get_contents($filePath);
        if ($imageData === false) {
            return false;
        }
        $finfo    = finfo_open(FILEINFO_MIME_TYPE);
        $mimeType = finfo_buffer($finfo, $imageData);
        finfo_close($finfo);
        return 'data:' . $mimeType . ';base64,' . base64_encode($imageData);
    }

    public function base64ToTempFile($base64)
    {
        try {
            if (strpos($base64, 'data:image/') === 0) {
                $parts = explode(',', $base64, 2);
                if (count($parts) !== 2) {
                    return false;
                }
                $base64 = $parts[1];
            }

            $imageData = base64_decode($base64, true);
            if ($imageData === false) {
                return false;
            }

            $tempDir = __DIR__ . '/assets/temp';
            if (! is_dir($tempDir)) {
                mkdir($tempDir, 0755, true);
            }

            $tempFile = $tempDir . '/face_match_' . uniqid() . '.jpg';
            if (file_put_contents($tempFile, $imageData) === false) {
                return false;
            }

            return $tempFile;
        } catch (\Exception $e) {
            return false;
        }
    }

    public function compare($img1, $img2, $threshold = null)
    {
        if (empty($img1) || empty($img2)) {
            return ['error' => 'Both image sources are required'];
        }

        DeepFace::log("compare() called with img1: $img1, img2: $img2, threshold: $threshold");
        self::startServerIfNeeded($this->apiUrl);
        $startTime = microtime(true);
        $result    = null;
        try {
            $url    = $this->apiUrl . '/verify';
            $fields = [
                'model_name' => 'VGG-Face',
                'img1'       => $this->prepareCurlFile($img1),
                'img2'       => $this->prepareCurlFile($img2),

            ];
            if ($threshold !== null) {
                $fields['threshold'] = $threshold;
            }
            $files = [

            ];
            DeepFace::log("Sending POST to $url with fields: " . json_encode($fields));
            $response = $this->curlPost($url, $fields, $files);
            DeepFace::log("Received response: $response");
            $result = json_decode($response, true);
        } catch (\Exception $e) {
            DeepFace::log("Error in compare(): " . $e->getMessage());
            $result = ['error' => $e->getMessage()];
        }
        $endTime = microtime(true);
        if (! isset($result['total_time_seconds'])) {
            $result['total_time_seconds'] = round($endTime - $startTime, 4);
        }
        return $result;
    }

    public function analyze($img)
    {
        if (empty($img)) {
            return ['error' => 'Image source is required'];
        }
        DeepFace::log("analyze() called with img: $img");
        self::startServerIfNeeded($this->apiUrl);
        $startTime = microtime(true);
        $result    = null;
        try {
            $url    = $this->apiUrl . '/analyze';
            $fields = [
                'actions'    => "['age', 'gender', 'emotion', 'race']",
                'model_name' => 'VGG-Face',
                'img'        => $this->prepareCurlFile($img),

            ];
            $files = [

            ];
            DeepFace::log("Sending POST to $url with fields: " . json_encode($fields));
            $response = $this->curlPost($url, $fields, $files);
            DeepFace::log("Received response: $response");
            $result = json_decode($response, true);
        } catch (\Exception $e) {
            DeepFace::log("Error in analyze(): " . $e->getMessage());
            $result = ['error' => $e->getMessage()];
        }
        $endTime = microtime(true);
        if (! isset($result['total_time_seconds'])) {
            $result['total_time_seconds'] = round($endTime - $startTime, 4);
        }
        return $result;
    }

    public static function compareImages($img1, $img2, $apiUrl = null, $threshold = null)
    {
        $instance = new self($apiUrl ?? self::$defaultApiUrl);
        return $instance->compare($img1, $img2, $threshold);
    }

    public static function analyzeImage($img, $apiUrl = null)
    {
        $instance = new self($apiUrl ?? self::$defaultApiUrl);
        return $instance->analyze($img);
    }

    private function prepareCurlFile($input)
    {
        // If it's a remote URL, download to temp file
        if (filter_var($input, FILTER_VALIDATE_URL)) {
            $tmpDir = __DIR__ . '/assets/temp';
            if (! is_dir($tmpDir)) {
                mkdir($tmpDir, 0755, true);
            }
            $tmp     = $tmpDir . '/face_match_url_' . uniqid() . '.jpg';
            $imgData = @file_get_contents($input);
            if ($imgData === false) {
                DeepFace::log("prepareCurlFile: failed to download remote URL $input");
                throw new \Exception('Failed to download remote image: ' . $input);
            }
            file_put_contents($tmp, $imgData);
            DeepFace::log("prepareCurlFile: downloaded remote URL $input to $tmp");
            return $tmp;
        }
        // If it's a file path
        if ($this->isFilePath($input)) {
            if (! file_exists($input)) {
                DeepFace::log("prepareCurlFile: file does not exist: $input");
                throw new \Exception('File does not exist: ' . $input);
            }
            DeepFace::log("prepareCurlFile: using file path $input");
            return $input;
        }
        // Otherwise, treat as base64
        $tmp = $this->base64ToTempFile($input);
        DeepFace::log("prepareCurlFile: created temp file $tmp from base64");
        if (! $tmp) {
            DeepFace::log("prepareCurlFile: failed to create temp file from base64");
            throw new \Exception('Invalid image data');
        }
        // Always return the file path for FastAPI
        return $tmp;
    }

    private function curlPost($url, $fields, $files)
    {
        DeepFace::log("curlPost: POST $url");
        $ch = curl_init();
        // Always send JSON data
        $jsonData = json_encode($fields);
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_POST, 1);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $jsonData);
        curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        $response = curl_exec($ch);
        if (curl_errno($ch)) {
            $error_msg = curl_error($ch);
            DeepFace::log("curlPost: Curl error: $error_msg");
            curl_close($ch);
            throw new \Exception('Curl error: ' . $error_msg);
        }
        curl_close($ch);
        DeepFace::log("curlPost: Response: $response");
        return $response;
    }

    private function parseOutput($output)
    {
        $matches = [];
        if (preg_match('/\{.*\}$/s', trim($output ?? ''), $matches)) {
            $json = $matches[0];
            $data = json_decode($json, true);
        } else {
            $data = null;
        }

        if (! $data) {
            throw new \Exception('Invalid or empty response from Python script: ' . $output);
        }

        if (isset($data['error'])) {
            throw new \Exception($data['error']);
        }

        return $data;
    }
}
