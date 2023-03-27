import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

class Extractor(object):
    def __init__(self, intrinsic_matrix):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = intrinsic_matrix
        self.Kinv = np.linalg.inv(intrinsic_matrix)

    def convert_cartesian_to_homogenous(self, cartesian_coords):
        homogenous_coords = np.concatenate([cartesian_coords, np.ones((cartesian_coords.shape[0], 1))], axis = 1)
        return homogenous_coords

    def normalize(self):
        return np.dot(self.Kinv, convert_cartesian_to_homogenous(pts).T).T[:,0:2]

    def denormalize(self):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1])))

    def extract(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = cv2.goodFeaturesToTrack(gray, 3000, qualityLevel=0.01, minDistance=3)

        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        # matching
        ret = []
        if self.last is not None:
          matches = self.bf.knnMatch(des, self.last['des'], k=2)
          for m,n in matches:
            if m.distance < 0.75*n.distance:
              kp1 = kps[m.queryIdx].pt
              kp2 = self.last['kps'][m.trainIdx].pt
              ret.append((kp1, kp2))

        # filter
        if len(ret) > 0:
          ret = np.array(ret)

          ret[:, 0, :] = self.normalize(ret[:,0,:])
          ret[:, 1, :] = self.normalize(ret[:,1,:])

          model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                  FundamentalMatrixTransform, 
                                  EssentialMatrixTransform
                                  min_samples=8,
                                  residual_threshold=0.005,
                                  max_trials=200)
          ret = ret[inliers]

        # return
        self.last = {'kps': kps, 'des': des}
        return ret
