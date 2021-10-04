import cv2
import imutils
import onnxruntime as ort
import warnings
import time
from ultra_onnx import faceDetector, scale
import argparse
from retinex_model.retinex import *
import json


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default='dereck.mp4', help='Enter input video')
# ap.add_argument("-o", "--output", default='../data/video_output/ultra_light_face.mp4', help="Path of Output video")
arg = vars(ap.parse_args())


warnings.filterwarnings('ignore')
face_detector_onnx = "./version-RFB-320.onnx"
face_detector = ort.InferenceSession(face_detector_onnx)


with open('config.json', 'r') as f:
    config = json.load(f)

cap = cv2.VideoCapture(arg['video'])
cnt = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

# results = cv2.VideoWriter(arg['output'],
#                            cv2.VideoWriter_fourcc(*'XVID'),
#                            10,
#                            size)

color = (255, 128, 0)
prev_frame_t = 0
new_frame_t = 0

while cap.isOpened():
    ret, frame = cap.read()
    # check ret to see if frame is there

    if ret:
        frame = imutils.resize(frame, width=500)
        frame1 = MSRCP(
            frame,
            config['sigma_list'],
            config['low_clip'],
            config['high_clip']
        )
        # frame1 = frame
        boxes, labels, probs = faceDetector(face_detector, frame1)
        # check if model detects face
        for i in range(boxes.shape[0]):
            box = scale(boxes[i, :])
            # conf = float(probs)
            # text = "{:.2f}%".format(conf * 100)
            cv2.rectangle(frame1, (box[0], box[1]), (box[2], box[3]), color, 4)
            # cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        if list(boxes):
            print(f"detected {cnt}")
            cnt += 1

        # calculate fps
        new_frame_t = time.time()
        fps = 1/(new_frame_t - prev_frame_t)
        prev_frame_t = new_frame_t
        fps = str(int(fps))
        fps = "FPS:- " + fps
        cv2.putText(frame1, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 255), 3, cv2.LINE_AA)

        # write video
        # results.write(frame)

        # show detected frame
        cv2.imshow('frame', frame1)

        # quit key
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    else:
        break


cap.release()
# results.release()
cv2.destroyAllWindows()
