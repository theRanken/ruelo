#!/usr/bin/env python3
import sys
import os
import json
import warnings
import logging

# Silence Python warnings and logs
warnings.filterwarnings('ignore', category=Warning)
logging.getLogger().setLevel(logging.ERROR)

# If third-party libraries are still logging (e.g. numpy), disable their loggers too
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL + 1)

from matchers.opencv import OpenCVFaceMatcher
from matchers.deepface import DeepFaceMatcher

def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"error": "Usage: python deepface_cli.py [--compare img1 img2 threshold] [--analyze img] [--method opencv|onnx]"}))
            sys.exit(1)

        matcher = DeepFaceMatcher()

        if sys.argv[1] == '--compare' and len(sys.argv) >= 4:
            image1_source = sys.argv[2]
            image2_source = sys.argv[3]
            threshold = float(sys.argv[4]) if len(sys.argv) > 4 else None
            result = matcher.compare_faces(image1_source, image2_source, threshold)
        elif sys.argv[1] == '--analyze' and len(sys.argv) >= 3:
            image_source = sys.argv[2]
            result = matcher.analyze_face(image_source)
        else:
            result = {"error": "Invalid command. Use --compare or --analyze"}

        # Output clean JSON result
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        # In case of any unexpected error
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
