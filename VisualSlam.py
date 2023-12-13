import cv2
import os
import datetime
import math
import numpy as np
from tqdm import tqdm

class SLAM():
    def __init__(self,Cmat):        
        # self.images = self._load_images(image_dir)
        self.orb = cv2.ORB.create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        # self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.K = Cmat
        self.first_process =  True
        # print("K: {}".format(self.K))
        
        
    def _form_transf(self, R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    
    def find_feature_points(self,img_now,img_prev):
        # Find the keypoints and descriptors with ORB
        # kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        # kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        kp1, des1 = self.orb.detectAndCompute(img_prev, None)
        kp2, des2 = self.orb.detectAndCompute(img_now, None)
        return kp1,kp2,des1,des2
    
    def find_feature_points_singe_img(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des
        
    
    def get_matches(self, kp1,kp2,des1,des2):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        

        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)
        # matches = self.bf.match(des1, des2)

        # Find the matches there do not have a to high distance
        good = []

        dist_threshold = 60
        try:
            for m, n in matches:
                # print("m:\n{}\nn:\n{}".format(m, n))
                x1, y1 = kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]
                x2, y2 = kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]

                dist = math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))

                if dist <= dist_threshold:
                    if m.distance < 0.8 * n.distance:
                        good.append(m)
        except ValueError:
            pass

        # draw_params = dict(matchColor = -1, # draw matches in green color
        #          singlePointColor = None,
        #          matchesMask = None, # draw only inliers
        #          flags = 2)

        # img3 = cv2.drawMatches(img_now, kp1, img_prev,kp2, good ,None,**draw_params)
        # cv2.imshow("image", img3)
        # key = cv2.waitKey(2)
        
        # # Check if the 'q' key is pressed (you can change 'q' to any key you prefer)
        # if key & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()  # Close the OpenCV window

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2, good
    
    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        time_start = datetime.datetime.now()
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)      # Adjust the threshold
        time_emat = datetime.datetime.now()
        time_diff = time_emat - time_start
        # print("Time taken (essentialMat): {}".format(time_diff))

        _, R, t, masks = cv2.recoverPose(E, q1, q2, self.K)
        time_recpose = datetime.datetime.now()
        time_diff = time_recpose - time_emat
        # print("Time taken (recoverPose): {}".format(time_diff))
        # Decompose the Essential matrix into R and t
        # if self.first_process == True:
        #     self.P = np.column_stack((self.K, np.zeros((3, 1))))
        #     self.first_process = False
        # print("P: {}".format(self.P))

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        time_formtransf = datetime.datetime.now()
        time_diff = time_formtransf - time_recpose
        # print("Time taken (form_transf): {}".format(time_diff))
        return transformation_matrix
    
   
    




