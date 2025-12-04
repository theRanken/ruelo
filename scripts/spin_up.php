<?php

$python = 'python';
$script = realpath(__DIR__ . '/../../src/scripts/df_service.py');
if (!$script || !file_exists($script)) {
    echo "Script not found: $script\n";
    exit(1);
}
$scriptDir = dirname($script);
$scriptBase = basename($script);
// Set working directory and properly quote script path
$cmd = "cd \"$scriptDir\" && $python \"$scriptBase\" > /dev/null 2>&1 &";
echo "CMD: $cmd\n";
$output = shell_exec($cmd);
echo "Output: $output\n";
echo "FastAPI DeepFace server started.\n";