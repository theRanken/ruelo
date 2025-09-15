<?php

/**
 * Script to manually stop the DeepFace FastAPI server.
 * This script can be used in any application that uses the DeepFace library.
 */

$apiUrl = 'http://127.0.0.1:4800';
$healthUrl = rtrim($apiUrl, '/') . '/health';

echo "Checking if FastAPI server is running at $healthUrl\n";
$health = @file_get_contents($healthUrl);
if ($health === false) {
    echo "FastAPI server is not running.\n";
    exit(0);
}

echo "FastAPI server is running. Attempting to stop it.\n";

if (strncasecmp(PHP_OS, 'WIN', 3) === 0) {
    // Windows: Use taskkill to kill python processes running uvicorn
    $cmd = 'taskkill /F /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq FastAPI"';
    exec($cmd, $output, $ret);
    if ($ret === 0) {
        echo "FastAPI server stopped successfully.\n";
    } else {
        echo "Failed to stop FastAPI server. You may need to manually kill the process.\n";
    }
} else {
    // Unix/Linux/macOS: Use pkill to kill processes containing 'uvicorn'
    $cmd = 'pkill -f uvicorn';
    exec($cmd, $output, $ret);
    if ($ret === 0) {
        echo "FastAPI server stopped successfully.\n";
    } else {
        echo "Failed to stop FastAPI server. You may need to manually kill the process.\n";
    }
}