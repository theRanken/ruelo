import os, sys, json, argparse
from deepface import DeepFace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Suppress TensorFlow warnings


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
        args.img1 = str(args.img1).strip()
        args.img2 = str(args.img2).strip()
        if not os.path.isfile(args.img1) or not os.path.isfile(args.img2):
            raise ValueError('Both image paths must be valid files')
        result = DeepFace.verify(img1_path=args.img1, img2_path=args.img2, detector_backend='retinaface', model_name='Facenet512')
    elif args.action == 'analyze':
        result = DeepFace.analyze(img_path=args.img1, actions=args.models)
    print(json.dumps({'success': True, 'result': result}))
except Exception as e:
    print(json.dumps({'success': False, 'error': str(e)}))
    sys.exit(1)
