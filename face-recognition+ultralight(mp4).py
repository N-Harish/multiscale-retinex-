'''
@author Dereck Jos and Harish Natarajan
Face Recognition Dlib
'''
import face_recognition
import cv2
import onnxruntime as ort
import numpy as np
import warnings
import time
from ultra_onnx import faceDetector, scale
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default=0, help='Enter input video')
args = vars(ap.parse_args())

warnings.filterwarnings('ignore')
face_detector_onnx = "./version-RFB-320.onnx"
face_detector = ort.InferenceSession(face_detector_onnx)

dereck_image = face_recognition.load_image_file("dereck.jpg")
dereck_face_encoding = face_recognition.face_encodings(dereck_image)[0]


harish_image = face_recognition.load_image_file("harish.jpg")
harish_face_encoding = face_recognition.face_encodings(harish_image)[0]

known_face_encodings = [
    dereck_face_encoding,
    harish_face_encoding
    ]

known_face_names = ["Dereck Jos", "Harish Natarajan"]

cap = cv2.VideoCapture(args["video"])

cnt = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

name=""

color = (255, 128, 0)
prev_frame_t = 0
new_frame_t = 0

# name = "Unknown"

f1 = 0
e1 = 0
while cap.isOpened():
    grabbed, frame = cap.read()
    # check ret to see if frame is there

    if grabbed:
        rgb_frame = frame[:, :, ::-1]
        boxes, labels, probs = faceDetector(face_detector, frame)
        # check if model detects face
        for i in range(boxes.shape[0]):
            box = scale(boxes[i, :])
            x,y,w,h = (box[0], box[1], box[2], box[3])
            face = rgb_frame[y:h, x:w]
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            try:
                face_locations = face_recognition.face_locations(face)
                face_encodings = face_recognition.face_encodings(face, face_locations)
                f1 = f1+1
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    name = "Unknown"
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]


            except:
                name = "Unknown"

            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, h + 40), (w, h), (0, 0, 255), -1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x, h + 20), font, 0.7, (255, 255, 255), 1)
            name = "Unknown"

        cnt += 1

        # calculate fps
        new_frame_t = time.time()
        fps = 1/(new_frame_t - prev_frame_t)
        prev_frame_t = new_frame_t
        fps = str(int(fps))
        fps = "FPS:- " + fps
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 255), 3, cv2.LINE_AA)

        # show detected frame
        cv2.imshow('frame', frame)

        # quit key
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    else:
        break

print(cnt)
print(f1)
print(e1)
cap.release()
cv2.destroyAllWindows()



