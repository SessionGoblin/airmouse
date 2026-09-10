"""Local integration check; never enables desktop input or saves camera video."""
import argparse
import time
import numpy as np
from airmouse.tracking import HandTracker
from airmouse.capture import Camera

parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, help='Opt in to four seconds of webcam processing')
args = parser.parse_args()
tracker = HandTracker()
camera = None
try:
    assert tracker.detect(np.zeros((480,640,3), dtype=np.uint8)) == []
    print('Model initialization and blank-frame inference: passed')
    if args.camera is not None:
        camera = Camera(args.camera)
        start = time.monotonic()
        count = 0
        while time.monotonic()-start < 4:
            item = camera.take()
            if item:
                tracker.detect(item[1])
                count += 1
            else:
                time.sleep(.005)
        assert count > 0, camera.error or 'No camera frames'
        print(f'Webcam inference: {count} frames processed')
finally:
    if camera:
        camera.close()
        assert not camera.thread.is_alive(), 'Camera failed to stop'
    tracker.close()
print('Resource cleanup: passed')
