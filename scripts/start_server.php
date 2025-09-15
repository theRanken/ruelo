<?php

/**
 * Script to manually start the DeepFace FastAPI server.
 * This script can be used in any application that uses the DeepFace library.
 */

$apiUrl = 'http://127.0.0.1:4800';
$healthUrl = rtrim($apiUrl, '/') . '/health';

echo "Checking FastAPI health at $healthUrl\n";
$health = @file_get_contents($healthUrl);
if ($health === false) {
    $python = 'python';
    $pythonScript = realpath(__DIR__ . '/../src/scripts/Python/df_service.py');
    echo "FastAPI not running. Attempting to start: $pythonScript\n";
    if (strncasecmp(PHP_OS, 'WIN', 3) === 0) {
        // Windows
        $cmd = "start /B \"FastAPI\" \"$python\" \"$pythonScript\"";
    } else {
        // Linux/macOS
        $cmd = "$python \"$pythonScript\" > /dev/null 2>&1 &";
    }
    exec($cmd, $output, $ret);
    echo "Executed command: $cmd | Return: $ret | Output: " . implode(' ', $output) . "\n";

    $start = time();
    while (true) {
        usleep(500000); // 0.5s
        $health = @file_get_contents($healthUrl);
        echo "Waiting for FastAPI health... Status: " . ($health !== false ? 'OK' : 'NOT OK') . "\n";
        if ($health !== false || (time() - $start) > 20) {
            break;
        }
    }
    if ($health === false) {
        echo "FastAPI server failed to start or is unreachable after 20 seconds.\n";
        exit(1);
    } else {
        echo "FastAPI server is up and healthy.\n";
    }
} else {
    echo "FastAPI server is already running.\n";
}

echo "DeepFace server started successfully.\n";