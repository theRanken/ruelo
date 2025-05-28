#!/usr/bin/env python3
import sys, json
from matchers.lib import MediaPipeFaceMatcher

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python deepface_cli.py [--compare img1 img2 threshold] [--analyze img]"}))
        sys.exit(1)
    
    matcher = MediaPipeFaceMatcher()
    
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
    
    print(json.dumps(result))

if __name__ == "__main__":
    main()