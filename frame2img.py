import cv2
import imutils


cap = cv2.VideoCapture("dereck.mp4")

cnt = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=500)
    if ret:
        cv2.imshow("frame", frame)
        cv2.imwrite(f"./data/frame_{cnt}.jpg", frame)
        cnt += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    else:
        break


cv2.destroyAllWindows()
cap.release()
