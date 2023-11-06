import pygame
import cv2
import os
import time
import math
import numpy as np
import torch

import matplotlib.pyplot as plt

from vis_nav_game import Player, Action

from test_inputs import test_inputs

from VisualSlam import SLAM
from plot_path import visualize_paths

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor, make_matching_plot_fast

START_STEP = 5
STEP_SIZE = 3




# SuperPoint / SuperGlue options
class SuperOpt():
    def __init__(self):
        self.nms_radius = 4
        self.keypoint_threshold = 0.005
        self.max_keypoints = 80

        self.superglue = 'indoor'
        self.sinkhorn_iterations = 20
        self.match_threshold = 0.3

        # self.show_keypoints = False


# Player class
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.relative_poses = [] 
        self.estimated_path = []
        self.cur_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float16)
        super(KeyboardPlayerPyGame, self).__init__()
                
        self.img_idx = 0  # Add a counter for image filenames
        self.last_save_time = time.time()  # Record the last time an image was saved
        self.all_fpv = []
        self.cum_turned_angle = 0
        self.prev_angle = 0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.super_opt = SuperOpt()

        self.super_config = {
            'superpoint': {
                'nms_radius': self.super_opt.nms_radius,
                'keypoint_threshold': self.super_opt.keypoint_threshold,
                'max_keypoints': self.super_opt.max_keypoints
            },
            'superglue': {
                'weights': self.super_opt.superglue,
                'sinkhorn_iterations': self.super_opt.sinkhorn_iterations,
                'match_threshold': self.super_opt.match_threshold,
            }
        }

        self.matching = Matching(self.super_config).eval().to(self.device)
        self.super_keys = ['keypoints', 'scores', 'descriptors']
        self.img_data_list = []
        self.img_tensor_list = []

        self.prev_img_tensor = None

        self.starting_step = START_STEP
        self.step_size = STEP_SIZE

        # self.tick_turn_rad = 0.042454
        self.tick_turn_rad = 0.0426

        self.orb = cv2.ORB.create(3000)

        self.r = 0
        self.theta = 0

        self.prev_reck_act = None

        self.testing_with_inputs = False
        self.test_input_idx = 0
        self.test_action_idx = 0

        self.action_list = []


    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }
        
    def pre_exploration(self) -> None:
        self.Cmat = self.get_camera_intrinsic_matrix()
        # print(self.Cmat)
        self.slam = SLAM(self.Cmat)

        # Save starting location
        self.estimated_path.append((self.cur_pose[0,3], self.cur_pose[2,3]))
       
    def act(self):

        if not self.testing_with_inputs:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.last_act = Action.QUIT
                    return Action.QUIT

                if event.type == pygame.KEYDOWN:
                    if event.key in self.keymap:
                        self.last_act |= self.keymap[event.key]
                    else:
                        self.show_target_images()
                if event.type == pygame.KEYUP:
                    if event.key in self.keymap:
                        self.last_act ^= self.keymap[event.key]

        else:
            if self.test_input_idx >= len(test_inputs):
                return Action.IDLE
                
            test_step = test_inputs[self.test_input_idx]

            self.last_act = Action.IDLE

            self.last_act |= test_step['actions']

            self.test_action_idx += 1
            if (self.test_action_idx >= test_step['steps']):
                self.test_action_idx = 0
                self.test_input_idx += 1
        
        return self.last_act

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        # data_dir = r"C:\Users\ifeda\ROB-GY-Computer-Vision\vis_nav_player"
        visualize_paths(self.estimated_path, "Visual Odometry",file_out="VO.html")
        
        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)
        # print(self.estimated_path)
        


    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()
        

    # Find pose via dead reckoning
    # Use polar coordinates to calculate vector from previous position to current position
    def find_pose_dead_reck(self):
        # If moving forward or backward, r = 1 (radius)
        if Action.FORWARD in self.last_act:
            self.r = 1
            self.prev_reck_act = Action.FORWARD
        elif Action.BACKWARD in self.last_act:
            self.r = -1
            self.prev_reck_act = Action.BACKWARD
        else:
            self.r = 0

        # If moving left or right, theta = 1 (angle)
        if Action.LEFT in self.last_act:
            self.theta = self.theta + self.tick_turn_rad
            self.prev_reck_act = Action.LEFT
        elif Action.RIGHT in self.last_act:
            # self.r = 0
            self.theta = self.theta - self.tick_turn_rad
            self.prev_reck_act = Action.RIGHT
        else:
            pass

        # Constrain theta between 0 and 2pi
        if self.theta >= (2 * math.pi):
            self.theta = self.theta - (2 * math.pi)
        elif self.theta < 0:
            self.theta = (2 * math.pi) + self.theta

        # Calculate x,y coordinates from r,theta
        x = self.r * math.cos(self.theta)
        y = self.r * math.sin(self.theta)

        # Update current pose (add vector to current pose)
        self.cur_pose[0,3] = self.cur_pose[0,3] + x
        self.cur_pose[2,3] = self.cur_pose[2,3] + y

        # print("r: {}, theta: {}, x: {}, y: {}, posex: {}, posey: {}".format(self.r, self.theta, x, y, self.cur_pose[0,3], self.cur_pose[2,3]))

        # Save current location
        self.estimated_path.append((self.cur_pose[0,3], self.cur_pose[2,3]))


    # Find feature points using SuperPoint
    def find_feature_points_superpoint(self, img):
        img_tensor = frame2tensor(img, self.device)
        img_data = self.matching.superpoint({'image': img_tensor})

        self.img_data_list.append(img_data)
        self.img_tensor_list.append(img_tensor)

        return img_data['keypoints'], img_data['descriptors']


    # Find feature matches using SuperGlue
    def find_feature_matches_superglue(self, img1_index, img2_index):
        img1_data = {k+'0':self.img_data_list[img1_index][k] for k in self.super_keys}
        img1_data['image0'] = self.img_tensor_list[img1_index]

        img2_data = {k+'1': self.img_data_list[img2_index][k] for k in self.super_keys}
        img2_data['image1'] = self.img_tensor_list[img2_index]

        pred = self.matching({**img1_data, **img2_data})
        kpts0 = img1_data['keypoints0'][0].cpu().numpy()
        kpts1 = img2_data['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        # confidence = pred['matching_scores0'][0].cpu().detach().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        # Get matching points (q1 for img1, q2 for img2)
        q1 = np.array(mkpts0)
        q2 = np.array(mkpts1)
        

        return q1, q2


    # Find pose
    def find_pose(self, q1, q2):
        # Get pose from SLAM class
        relative_pose  = self.slam.get_pose(q1, q2)
        if self.last_act == Action.FORWARD or self.last_act == Action.BACKWARD:
            # relative_pose[:,:3] = 1
            relative_pose[:3,:3] = np.eye(3)
            pass
        elif self.last_act == Action.LEFT or self.last_act == Action.RIGHT:
            relative_pose[:,3] = [0,0,0,1]
        elif self.last_act == Action.IDLE:
            relative_pose[:,:] = np.eye(4)
            
        relative_pose = np.nan_to_num(relative_pose, neginf=0, posinf=0)
          
        # Save last x,z coordinates
        prev_xz = (self.cur_pose[0,3], self.cur_pose[2,3])
        prev_rot = self.cur_pose[:3,:3]

        # Calculate new pose from relative pose (transformation matrix)
        self.cur_pose = np.matmul(self.cur_pose, np.linalg.inv(relative_pose))
        
        # print("curr pose:\n{}".format(cur_pose))

        # rotation_matrix = self.cur_pose[:3, :3]

        # Calculate the rotation about the y-axis (pitch) using the arctan2 function
        # pitch = np.arctan2(rotation_matrix[2, 0], np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
        
        # Calculate the rotation about the y-axis (pitch) in radians using the arctan2 function
        # Convert to degrees if needed
        # yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        # yaw_degrees = np.degrees(yaw)
        # turned_angle = self.prev_angle - yaw_degrees
        # # turned_angle = self.prev_angle - pitch_degrees
        # self.cum_turned_angle += turned_angle
        
        # print("cum angle", self.cum_turned_angle)
        # print("cum angle", self.cum_turned_angle)
        # print("cum angle", self.cum_turned_angle)
        # print("cum angle", self.cum_turned_angle)
        
        
        # self.prev_angle = yaw_degrees       
        # self.prev_angle = pitch_degrees
        # If not moving forward or backward, ignore the translation vector
        # Translation vector seems to be normalized to 1 from decomposeEssentialMat()
        # See: https://answers.opencv.org/question/66839/units-of-rotation-and-translation-from-essential-matrix/
        if (self.last_act != Action.FORWARD) and (self.last_act != Action.BACKWARD):
            self.cur_pose[0,3] = prev_xz[0]
            self.cur_pose[2,3] = prev_xz[1]
        if (self.last_act != Action.LEFT) and (self.last_act != Action.RIGHT):
            self.cur_pose[:3,:3] = prev_rot
        
        print(self.cur_pose[0,3],self.cur_pose[2,3],self.cur_pose[0,3]-prev_xz[0],self.cur_pose[2,3]-prev_xz[1])
        print(self.cur_pose[0,3],self.cur_pose[2,3],self.cur_pose[0,3]-prev_xz[0],self.cur_pose[2,3]-prev_xz[1])
        print(self.cur_pose[0,3],self.cur_pose[2,3],self.cur_pose[0,3]-prev_xz[0],self.cur_pose[2,3]-prev_xz[1])
        print(self.cur_pose[0,3],self.cur_pose[2,3],self.cur_pose[0,3]-prev_xz[0],self.cur_pose[2,3]-prev_xz[1])
        print(self.cur_pose[0,3],self.cur_pose[2,3],self.cur_pose[0,3]-prev_xz[0],self.cur_pose[2,3]-prev_xz[1])
        print(self.cur_pose[0,3],self.cur_pose[2,3],self.cur_pose[0,3]-prev_xz[0],self.cur_pose[2,3]-prev_xz[1])
        print(self.cur_pose[0,3],self.cur_pose[2,3],self.cur_pose[0,3]-prev_xz[0],self.cur_pose[2,3]-prev_xz[1])
        print(self.cur_pose[0,3],self.cur_pose[2,3],self.cur_pose[0,3]-prev_xz[0],self.cur_pose[2,3]-prev_xz[1])
        
        
        
        x_diff = self.cur_pose[0,3]-prev_xz[0]
        y_diff = self.cur_pose[2,3]-prev_xz[1]
        
        if abs(x_diff) > abs(y_diff):
        # If the difference on the x-axis is greater, only update the x-axis position
            self.cur_pose[2, 3] = prev_xz[1]
        else:
        # If the difference on the y-axis is greater or equal, only update the y-axis position
            self.cur_pose[0, 3] = prev_xz[0]
            
            
        # Save current location
        self.estimated_path.append((self.cur_pose[0,3], self.cur_pose[2,3]))
        # print(self.estimate)
        self.action_list.append(self.last_act)

        return self.cur_pose


    # Process image
    def process_image_super_glue(self, fpv):
        state = self.get_state()
        if state is None:
            return None
        
        step = state[2]

        if self.last_act == Action.IDLE:
            return True
        
        
        # If past starting step (to avoid static) and on a set interval (self.step_size)
        if (step > self.starting_step) and ((step % self.step_size) == 0):
            fpv_gray = cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY)
            
            # Find feature points
            keypts, desc = self.find_feature_points_superpoint(fpv_gray)

            # If more than one image processed (index >= 1)
            if self.img_idx >= 1:

                # Find feature matches between prev processed image and current image
                q1, q2 = self.find_feature_matches_superglue(self.img_idx-1, self.img_idx)

                pose = self.find_pose(q1, q2)

            # Increment index of processed images
            self.img_idx += 1

        return True
    
    
    # Find feature points using SuperPoint___delete this soon
    def find_feature_points_superpoint(self, img):
        img_tensor = frame2tensor(img, self.device)
        img_data = self.matching.superpoint({'image': img_tensor})

        self.img_data_list.append(img_data)
        self.img_tensor_list.append(img_tensor)

        return img_data['keypoints'], img_data['descriptors']
    
    
     
    # Process image
    def process_image_orb(self, fpv):
        state = self.get_state()
        if state is None:
            return None
        
        step = state[2]

        if self.last_act == Action.IDLE:
            return True
        
        
        # If past starting step (to avoid static) and on a set interval (self.step_size)
        if (step > self.starting_step) and ((step % self.step_size) == 0):
            fpv_gray = cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY)
            alpha = 2.0
            beta = 0.0
            fpv_gray = cv2.convertScaleAbs(fpv_gray, alpha=alpha, beta=beta)
            self.img_data_list.append(fpv_gray)
            
            # If more than one image processed (index >= 1)
            if self.img_idx >= 1:

                # Find feature matches between prev processed image and current image
                # Find feature points
                
                img_now = self.img_data_list[self.img_idx]
                img_prev = self.img_data_list[self.img_idx-1]
                kp1,kp2,des1,des2 = self.slam.find_feature_points(img_now,img_prev)
                q1,q2,good = self.slam.get_matches(kp1,kp2,des1,des2)
                draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

                img3 = cv2.drawMatches(img_now, kp1, img_prev,kp2, good ,None,**draw_params)
                cv2.imshow("image", img3)
                key = cv2.waitKey(2)
        
                # Check if the 'q' key is pressed (you can change 'q' to any key you prefer)
                if key & 0xFF == ord('q'):
                    cv2.destroyAllWindows()  # Close the OpenCV window
                
                if(len(q1)>=5):
                    pose = self.find_pose(q1, q2)

            # Increment index of processed images
            self.img_idx += 1

        return True


    # See (function used by game)
    def see(self, fpv):
        
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv
                   
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
       
        # Find pose via dead reckoning
        # self.find_pose_dead_reck()

        # Process image: find feature points, match feature points, get pose
        # ret = self.process_image_super_glue(fpv)
        
        ret = self.process_image_orb(fpv)
        
    

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
