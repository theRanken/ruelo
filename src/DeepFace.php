<?php

namespace DeepFacePHP;

class DeepFace
{
    private $pythonPath;
    private $scriptPath;

    public function __construct($pythonPath = 'python', $scriptPath = __DIR__ . '/scripts/Python/deepface_cli.py')
    {
        $this->pythonPath = $pythonPath;
        $this->scriptPath = $scriptPath;
    }

    public function compare($img1, $img2)
    {
        $cmd = escapeshellcmd("{$this->pythonPath} {$this->scriptPath} --action verify --img1 " . escapeshellarg($img1) . " --img2 " . escapeshellarg($img2));
        $output = shell_exec($cmd);
        return $this->parseOutput($output);
    }

    public function analyze($img, $models = [])
    {
        $modelsArg = '';
        if (!empty($models)) {
            $modelsArg = '--models ' . implode(' ', array_map('escapeshellarg', $models));
        }
        $cmd = escapeshellcmd("{$this->pythonPath} {$this->scriptPath} --action analyze --img1 " . escapeshellarg($img) . " $modelsArg");
        $output = shell_exec($cmd);
        return $this->parseOutput($output);
    }

    public function find($img, $dbPath, $model = 'VGG-Face', $distanceMetric = 'cosine', $threshold = 0.4)
    {
        $cmd = escapeshellcmd("{$this->pythonPath} {$this->scriptPath} --action find --img1 " . escapeshellarg($img) . " --db_path " . escapeshellarg($dbPath) . " --model " . escapeshellarg($model) . " --distance_metric " . escapeshellarg($distanceMetric) . " --threshold " . escapeshellarg($threshold));
        $output = shell_exec($cmd);
        return $this->parseOutput($output);
    }

    public static function compareImages($img1, $img2, $pythonPath = 'python', $scriptPath = __DIR__ . '/../deepface_cli.py')
    {
        $instance = new self($pythonPath, $scriptPath);
        return $instance->compare($img1, $img2);
    }

    public static function analyzeImage($img, $models = [], $pythonPath = 'python', $scriptPath = __DIR__ . '/../deepface_cli.py')
    {
        $instance = new self($pythonPath, $scriptPath);
        return $instance->analyze($img, $models);
    }

    private function parseOutput($output)
    {
        $data = json_decode($output ?? '', true);
        if (!$data || !$data['success']) {
            throw new \Exception($data['error'] ?? 'Unknown error');
        }
        return $data['result'];
    }
}
