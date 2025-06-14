<?php

namespace Ruelo;

class DeepFace
{
    private $pythonPath;
    private $scriptPath;

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
        if (!file_exists($filePath)) {
            return false;
        }
        $imageData = file_get_contents($filePath);
        if ($imageData === false) {
            return false;
        }
        $finfo = finfo_open(FILEINFO_MIME_TYPE);
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
            if (!is_dir($tempDir)) {
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

    public function __construct($pythonPath = 'python', $scriptPath = __DIR__ . "/scripts/Python/deepface_cli.py")
    {
        $this->pythonPath = $pythonPath;
        $this->scriptPath = $scriptPath;

        if (!file_exists($this->scriptPath)) {
            throw new \Exception("Script not found: {$this->scriptPath}");
        }
    }

    public function compare($img1, $img2, $threshold = null)
    {
        if (empty($img1) || empty($img2)) {
            return ['error' => 'Both image sources are required'];
        }

        $tempFiles = [];

        if (!$this->isFilePath($img1)) {
            $img1 = $this->base64ToTempFile($img1);
            if (!$img1)
                return ['error' => 'Invalid image 1 data'];
            $tempFiles[] = $img1;
        }

        if (!$this->isFilePath($img2)) {
            $img2 = $this->base64ToTempFile($img2);
            if (!$img2)
                return ['error' => 'Invalid image 2 data'];
            $tempFiles[] = $img2;
        }

        $thresholdArg = '';
        if ($threshold !== null && $threshold !== '') {
            $thresholdArg = escapeshellarg($threshold);
        }

        $cmd = sprintf(
            'set TF_CPP_MIN_LOG_LEVEL=3 && %s %s --compare %s %s %s 2>&1',
            escapeshellarg($this->pythonPath),
            escapeshellarg($this->scriptPath),
            escapeshellarg($img1),
            escapeshellarg($img2),
            $thresholdArg
        );

        putenv('TF_CPP_MIN_LOG_LEVEL=3');

        $startTime = microtime(true);
        $output = shell_exec($cmd);
        $endTime = microtime(true);

        foreach ($tempFiles as $file) {
            @unlink($file);
        }

        if ($output === null) {
            return ['error' => 'Failed to execute Python script'];
        }

        $result = $this->parseOutput($output);
        
        if (!isset($result['total_time_seconds'])) {
            $result['total_time_seconds'] = round($endTime - $startTime, 4);
        }

        return $result;
    }

    public function analyze($img)
    {
        if (empty($img)) {
            return ['error' => 'Image source is required'];
        }

        $tempFile = null;
        if (!$this->isFilePath($img)) {
            $img = $this->base64ToTempFile($img);
            if (!$img)
                return ['error' => 'Invalid image data'];
            $tempFile = $img;
        }

        $cmd = sprintf(
            'set TF_CPP_MIN_LOG_LEVEL=3 && %s %s --analyze %s 2>&1',
            escapeshellarg($this->pythonPath),
            escapeshellarg($this->scriptPath),
            escapeshellarg($img)
        );

        putenv('TF_CPP_MIN_LOG_LEVEL=3');

        $startTime = microtime(true);
        $output = shell_exec($cmd);
        $endTime = microtime(true);

        if ($tempFile) {
            @unlink($tempFile);
        }

        if ($output === null) {
            return ['error' => 'Failed to execute Python script'];
        }

        $result = $this->parseOutput($output);
        if (!isset($result['total_time_seconds'])) {
            $result['total_time_seconds'] = round($endTime - $startTime, 4);
        }

        return $result;
    }

    public static function compareImages($img1, $img2, $pythonPath = 'python', $scriptPath = __DIR__ . "/scripts/Python/deepface_cli.py", $threshold = null)
    {
        $instance = new self($pythonPath, $scriptPath);
        return $instance->compare($img1, $img2, $threshold);
    }

    public static function analyzeImage($img, $pythonPath = 'python', $scriptPath = __DIR__ . "/scripts/Python/deepface_cli.py")
    {
        $instance = new self($pythonPath, $scriptPath);
        return $instance->analyze($img);
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

        if (!$data) {
            throw new \Exception('Invalid or empty response from Python script: ' . $output);
        }

        if (isset($data['error'])) {
            throw new \Exception($data['error']);
        }

        return $data;
    }
}