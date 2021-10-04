'''
@author Dereck Jos
Face Recognition Dlib

'''
import face_recognition
import cv2
import imutils
import numpy as np
import pickle


video_capture = cv2.VideoCapture(1)

output = cv2.VideoWriter('face.mp4', cv2.VideoWriter_fourcc(*'XVID'), 5, (640,480))

dereck_image = face_recognition.load_image_file("dereck.jpg")
dereck_face_encoding = face_recognition.face_encodings(dereck_image)[0]

known_face_encodings = [
    dereck_face_encoding
    ]

known_face_names = ["Dereck Jos"]

while (video_capture.isOpened()):

    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]


    face_locations = face_recognition.face_locations(rgb_frame)
  
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

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

       
        cv2.rectangle(frame, (left, bottom), (right, top), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom+40), (right, bottom), (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left, bottom + 20), font, 0.7, (255, 255, 255), 1)


    output.write(frame)

    cv2.imshow('Video', frame)

 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()