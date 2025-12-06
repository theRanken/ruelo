<?php

namespace Ruelo;

class DeepFace
{
    /**
     * Base URL of the Python DeepFace API.
     * @var string
     */
    protected static string $apiUrl = 'http://127.0.0.1:4800';

    /**
     * Has the server health check / auto-start been attempted for this PHP process?
     * @var bool
     */
    protected static bool $serverChecked = false;

    /**
     * Enable or disable logging.
     * @var bool
     */
    public static bool $debug = false;

    // ─────────────────────────────────────────
    // Config
    // ─────────────────────────────────────────

    /**
     * Set the base URL for the DeepFace API (e.g. from env).
     */
    public static function setApiUrl(string $apiUrl): void
    {
        self::$apiUrl = rtrim($apiUrl, '/');
    }

    // ─────────────────────────────────────────
    // Public STATIC API (drop-in)
    // ─────────────────────────────────────────

    /**
     * Compare two faces.
     *
     * @param string      $img1      Path, URL or base64
     * @param string      $img2      Path, URL or base64
     * @param float|null  $threshold Optional threshold
     * @return array
     */
    public static function compare($img1, $img2, $threshold = 0.6): array
    {
        if (empty($img1) || empty($img2)) {
            return ['error' => 'Both image sources are required'];
        }

        self::ensureServerRunning();

        $start = microtime(true);

        try {
            $payload = [
                'img1'       => self::normalizeImageInput($img1),
                'img2'       => self::normalizeImageInput($img2),
                'model_name' => 'Facenet512', // faster than Facenet512
            ];

            if ($threshold !== null) {
                $payload['threshold'] = (float) $threshold;
            }

            $response = self::postJson('/verify', $payload);
        } catch (\Throwable $e) {
            $response = ['error' => $e->getMessage()];
        }

        $response['total_time_seconds'] = round(microtime(true) - $start, 4);

        return $response;
    }

    /**
     * Analyze one face (age, gender, emotion, race).
     *
     * @param string $img Path, URL or base64
     * @return array
     */
    public static function analyze($img): array
    {
        if (empty($img)) {
            return ['error' => 'Image source is required'];
        }

        self::ensureServerRunning();

        $start = microtime(true);

        try {
            $payload = [
                'img'     => self::normalizeImageInput($img),
                'actions' => ['age', 'gender', 'emotion', 'race'],
            ];

            $response = self::postJson('/analyze', $payload);
        } catch (\Throwable $e) {
            $response = ['error' => $e->getMessage()];
        }

        $response['total_time_seconds'] = round(microtime(true) - $start, 4);

        return $response;
    }

    // ─────────────────────────────────────────
    // Internal STATIC helpers (replacing old ones)
    // ─────────────────────────────────────────

    /**
     * Ensure the Python server is running.
     * Only checks / tries to spawn ONCE per PHP process.
     */
    protected static function ensureServerRunning(): void
    {
        if (self::$serverChecked) {
            return;
        }
        self::$serverChecked = true;

        $healthUrl = self::$apiUrl . '/health';

        // quick health check
        $context = stream_context_create([
            'http' => [
                'timeout' => 0.5,
            ],
        ]);

        $health = @file_get_contents($healthUrl, false, $context);
        if ($health !== false) {
            self::log('DeepFace API already running.');
            return;
        }

        // Optional: try to start the server in the background
        // Adjust the path to df_service.py as needed
        $python = PHP_OS_FAMILY === 'Windows' ? 'python' : 'python3';
        $script = __DIR__ . DIRECTORY_SEPARATOR . 'scripts/df_service.py';

        if (!file_exists($script)) {
            self::log("df_service.py not found at: {$script}");
            return;
        }

        if (PHP_OS_FAMILY === 'Windows') {
            // start /B hides window
            pclose(popen("start /B {$python} " . escapeshellarg($script) . " 2>&1", "r"));
        } else {
            exec($python . ' ' . escapeshellarg($script) . ' > /dev/null 2>&1 &');
        }

        self::log('DeepFace API started, waiting for health...');

        // Wait up to 5 seconds for /health to become OK
        $start = microtime(true);
        do {
            usleep(250000); // 0.25s
            $health = @file_get_contents($healthUrl, false, $context);
            if ($health !== false) {
                self::log('DeepFace API became healthy.');
                return;
            }
        } while (microtime(true) - $start < 30);

        self::log('DeepFace API did not become healthy within 5s.');
    }

    /**
     * Normalize image input to a data URL (base64) / URL / path string.
     *
     * @param string $source
     * @return string
     */
    protected static function normalizeImageInput(string $source): string
    {
        $source = trim($source);

        // Already a data URL
        if (stripos($source, 'data:image') === 0) {
            return $source;
        }

        // Looks like base64 (rough check: long base64-ish string)
        $clean = str_replace(["\r", "\n"], '', $source);
        if (strlen($clean) > 100 && preg_match('/^[A-Za-z0-9+\/]+=*$/', $clean)) {
            return $source;
        }

        // URL - let Python fetch it
        if (stripos($source, 'http://') === 0 || stripos($source, 'https://') === 0) {
            return $source;
        }

        // Local file path -> convert to base64 data URL
        if (is_file($source)) {
            $data = file_get_contents($source);
            if ($data === false) {
                throw new \RuntimeException("Unable to read image file: {$source}");
            }

            $mime = 'image/jpeg';
            if (function_exists('finfo_open')) {
                $finfo = finfo_open(FILEINFO_MIME_TYPE);
                if ($finfo) {
                    $detected = finfo_file($finfo, $source);
                    if ($detected) {
                        $mime = $detected;
                    }
                    finfo_close($finfo);
                }
            }

            $b64 = base64_encode($data);
            return "data:{$mime};base64,{$b64}";
        }

        // Fallback: just return the string and let Python try (it can handle paths too)
        return $source;
    }

    /**
     * POST JSON to the DeepFace API and decode the response.
     */
    protected static function postJson(string $endpoint, array $payload): array
    {
        set_time_limit(0);
        
        $url = self::$apiUrl . $endpoint;

        $ch = curl_init($url);
        if ($ch === false) {
            throw new \RuntimeException('Unable to initialize cURL');
        }

        $json = json_encode($payload);

        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST           => true,
            CURLOPT_HTTPHEADER     => ['Content-Type: application/json'],
            CURLOPT_POSTFIELDS     => $json,
            CURLOPT_TIMEOUT        => 0,
            CURLOPT_CONNECTTIMEOUT => 60,
        ]);

        $response = curl_exec($ch);
        if ($response === false) {
            $err = curl_error($ch);
            curl_close($ch);
            throw new \RuntimeException('cURL error: ' . $err);
        }

        $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        $decoded = json_decode($response, true);
        if (!is_array($decoded)) {
            throw new \RuntimeException(
                'Invalid JSON from DeepFace API: ' . substr($response, 0, 200)
            );
        }

        if ($status >= 400) {
            $msg = $decoded['error'] ?? $response;
            $trace = $decoded['traceback'] ?? '';
            throw new \RuntimeException("DeepFace API error ({$status}): " . $msg . "\n" . $trace);
        }

        set_time_limit(30);

        return $decoded;
    }

    /**
     * Simple file logger (optional).
     */
    protected static function log(string $message): void
    {
        if (!self::$debug) {
            return;
        }

        $logDir = __DIR__ . DIRECTORY_SEPARATOR . 'logs';
        if (!is_dir($logDir)) {
            @mkdir($logDir, 0777, true);
        }

        $logFile = $logDir . DIRECTORY_SEPARATOR . 'deepface.log';
        $line    = '[' . date('Y-m-d H:i:s') . '] ' . $message . PHP_EOL;
        @file_put_contents($logFile, $line, FILE_APPEND);
    }
}