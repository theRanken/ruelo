#!/usr/bin/env python3
import sys
import os

# Suppress all stderr output (must happen before any other stderr-using code)
sys.stderr = open(os.devnull, 'w')

# Suppress TensorFlow and other noisy logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

import json
import warnings
import logging

# Silence Python warnings and logs
warnings.filterwarnings('ignore', category=Warning)
logging.getLogger().setLevel(logging.ERROR)

# If third-party libraries are still logging (e.g. numpy), disable their loggers too
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL + 1)

from matchers.onnx_lib import OnnxFaceMatcher
from matchers.opencv_lib import OpenCVFaceMatcher

def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"error": "Usage: python deepface_cli.py [--compare img1 img2 threshold] [--analyze img] [--method opencv|onnx]"}))
            sys.exit(1)

        # Get the method from command line or default to opencv
        method = 'onnx' if '--method=onnx' in sys.argv else 'opencv'
        matcher = OpenCVFaceMatcher() if method == 'opencv' else OnnxFaceMatcher()
        
        # Remove method argument if it exists
        sys.argv = [arg for arg in sys.argv if not arg.startswith('--method=')]

        if sys.argv[1] == '--compare' and len(sys.argv) >= 4:
            image1_source = sys.argv[2]
            image2_source = sys.argv[3]
            threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
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
