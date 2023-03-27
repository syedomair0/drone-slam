import numpy as np
import cv2 
import numpy as np

from extractor import Extractor
#import open3d

cap = cv2.VideoCapture('../../drone-capture.mp4')
orb = cv2.ORB_create()
H, W = (2160//4, 3840//4)

F = 1
K = np.array([[F,0,W//2], [0,F,H//2], [0,0,1]]) #intrinsic matrix
extractor = Extractor(K)

def process_frame(frame):
    #H, W, layers = frame.shape
    frame = cv2.resize(frame, (W, H))

    matches = extractor.extract(frame)
    print("%d matches" % (len(matches)))

    for pt1, pt2 in matches:
        u1,v1 = extractor.denormalize(pt1)
        u2,v2 = extractor.denormalize(pt2)
        cv2.circle(frame, (u1,v1), color=(255,255,0), radius=3)
        cv2.line(frame, (u1,v1), (u2,v2), color=(255, 0, 255))

    return frame

if __name__ == "__main__":
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = process_frame(frame)
        cv2.imshow('frame', frame)


        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
