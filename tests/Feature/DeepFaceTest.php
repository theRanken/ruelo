<?php

use Ruelo\DeepFace;

beforeEach(function () {
    $this->deepFace = new DeepFace();
});

it('returns error when comparing with missing images', function () {
    $result = $this->deepFace->compare('', '');
    expect($result)->toBeArray();
    expect($result)->toHaveKey('error');
});

it('returns error when image files do not exist', function () {
    $result = $this->deepFace->compare('nonexistent1.jpg', 'nonexistent2.jpg');
    expect($result)->toBeArray();
    expect($result)->toHaveKey('error');
});

it('successfully compares two valid images and returns expected keys', function () {
    // Provide paths to two valid test images in the project or base64 strings
    $img1 = __DIR__ . '/../../src/assets/logo.jpg';
    $img2 = __DIR__ . '/../../src/assets/logo.jpg';

    $result = $this->deepFace->compare($img1, $img2);

    expect($result)->toBeArray();
    expect($result)->toHaveKey('status');
    expect($result['status'])->toBe('success');
    expect($result)->toHaveKey('verified');
    expect($result)->toHaveKey('confidence');
    expect($result)->toHaveKey('threshold');
});

it('returns error when analyzing with missing image', function () {
    $result = $this->deepFace->analyze('');
    expect($result)->toBeArray();
    expect($result)->toHaveKey('error');
});

it('successfully analyzes a valid image and returns expected keys', function () {
    $img = __DIR__ . '/../../src/assets/logo.jpg';

    $result = $this->deepFace->analyze($img);

    expect($result)->toBeArray();
    expect($result)->toHaveKey('status');
    expect($result['status'])->toBe('success');
    expect($result)->toHaveKey('dominant_emotion');
    expect($result)->toHaveKey('emotion');
    expect($result)->toHaveKey('face_confidence');
});
