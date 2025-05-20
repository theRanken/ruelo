import sys, json, argparse
from deepface import DeepFace

# Argument parsing
parser = argparse.ArgumentParser(description='DeepFace CLI for PHP integration')
parser.add_argument('--action', choices=['verify', 'analyze'], required=True, help='Action to perform')
parser.add_argument('--img1', required=True, help='Path to the first image')
parser.add_argument('--img2', help='Path to the second image (for verify)')
parser.add_argument('--models', nargs='*', help='Models to use for analysis')

args = parser.parse_args()

try:
    if args.action == 'verify':
        if not args.img2:
            raise ValueError('img2 is required for verify action')
        result = DeepFace.verify(img1_path=args.img1, img2_path=args.img2, detector_backend = 'retinaface', model='Facenet512')
    elif args.action == 'analyze':
        result = DeepFace.analyze(img_path=args.img1, actions=args.models)
    print(json.dumps({'success': True, 'result': result}))
except Exception as e:
    print(json.dumps({'success': False, 'error': str(e)}))
    sys.exit(1)
