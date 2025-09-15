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
    // Windows: Find PID of process listening on port 4800 and kill it
    $cmd = 'netstat -ano | findstr :4800';
    exec($cmd, $output, $ret);
    if ($ret === 0 && !empty($output)) {
        // Extract PID from the output (last column)
        $line = trim($output[0]);
        $parts = preg_split('/\s+/', $line);
        $pid = end($parts);
        if (is_numeric($pid)) {
            $killCmd = "taskkill /PID $pid /F";
            exec($killCmd, $killOutput, $killRet);
            if ($killRet === 0) {
                echo "FastAPI server stopped successfully (PID: $pid).\n";
                // Wait a bit for port to be released
                sleep(2);
            } else {
                echo "Failed to stop FastAPI server. You may need to manually kill the process.\n";
            }
        } else {
            echo "Could not determine PID from netstat output.\n";
        }
    } else {
        echo "No process found listening on port 4800.\n";
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