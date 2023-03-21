import numpy as np
import cv2 as cv
import open3d

cap = cv.VideoCapture('../drone-capture.mp4')
orb = cv.ORB_create()


while cap.isOpened():
    ret, frame = cap.read()
    height, width, layers = frame.shape
    frame = cv.resize(frame, (width//2, height//2))

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    kp, des = orb.detectAndCompute(frame, None)
    frame = cv.drawKeypoints(frame, kp, None, color=(0,255,255), flags=0)

    print(kp[0].pt, len(kp))
    print(des, des.shape)

    break

    #cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
