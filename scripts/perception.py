import numpy as np
from scipy.ndimage import filters
  
import cv2
import rospy
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from open3d.visualization import draw_plotly
from sensor_msgs.msg import CompressedImage, PointCloud2
from sensor_msgs import point_cloud2 as pc2

from ultralytics import YOLO

class Perception:
    def __init__(self):
        self.xyxy = None

    def TF_camera_to_base(self):
        rpy_base_to_hand = [-3.141583, 0.001902, 0.000062]
        translation_base_to_hand = [0.307027, 0, 0.530901]
        rotation_base_to_hand = R.from_euler('xyz', rpy_base_to_hand).as_matrix()

        rpy_hand_to_camera = [0, -1.35, 0]
        translation_hand_to_camera = [-0.115, 0.056, 0.018]
        rotation_hand_to_camera = R.from_euler('xyz', rpy_hand_to_camera).as_matrix()

        T_base_to_hand = np.eye(4)
        T_base_to_hand[:3, :3] = rotation_base_to_hand
        T_base_to_hand[:3, 3] = translation_base_to_hand
        T_hand_to_camera = np.eye(4)
        T_hand_to_camera[:3, :3] = rotation_hand_to_camera
        T_hand_to_camera[:3, 3] = translation_hand_to_camera

        T_base_to_camera = np.dot(T_base_to_hand, T_hand_to_camera)
        print("Transformation Matrix from Base to Camera:\n", T_base_to_camera)
        T_camera_to_base = np.linalg.inv(T_base_to_camera)
        print("Transformation Matrix from Camera to Base:\n", T_camera_to_base)

        return T_camera_to_base

    def calibrate_pc(self, point_cloud_np, T_camera_to_base):
        points_homogeneous = np.hstack((point_cloud_np, np.ones((point_cloud_np.shape[0], 1))))
        calibrated_points_homogeneous = np.dot(T_camera_to_base, points_homogeneous.T).T
        calibrated_points = calibrated_points_homogeneous[:, :3]

        return calibrated_points

    def callback_rgb(self, data):
        np_arr = np.frombuffer(data.data, np.uint8)
        rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
   
        # save_path = "/opt/ros_ws/src/perception/dataset/test1.jpg"
        # cv2.imwrite(save_path, rgb_image)
        # cv2.imshow('cv_img', rgb_image)

        model = YOLO("/opt/ros_ws/src/perception/model/best.pt")  # Load a trained model
        source = rgb_image
        results = model(source)                                     # return a list of Results objects

        for result in results:
            boxes = result.boxes                                    # Boxes object for bounding box outputs
            # print(boxes.xywh)
            self.xyxy = boxes.xyxy
            print(self.xyxy)
            masks = result.masks                                    # Masks object for segmentation masks outputs
            keypoints = result.keypoints                            # Keypoints object for pose outputs
            probs = result.probs                                    # Probs object for classification outputs
            obb = result.obb                                        # Oriented boxes object for OBB outputs
            result.save("/opt/ros_ws/src/perception/test_images/result0.jpg")

    def callback_pc(self, data):
        pc_data = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
        points = list(pc_data)
        point_cloud_np = np.array(points)

        T_camera_to_base = self.TF_camera_to_base()
        calibrated_points = self.calibrate_pc(point_cloud_np, T_camera_to_base)

        # z_threshold = 0.03  #[0.7,1]
        # filtered_points = point_cloud_np[(calibrated_points[:, 2] <= z_threshold)]
        
        calibrated_pc = o3d.geometry.PointCloud()
        calibrated_pc.points = o3d.utility.Vector3dVector(calibrated_points)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        coordinate_frame.transform(T_camera_to_base)
        
        o3d.visualization.draw_geometries([calibrated_pc, coordinate_frame])
        # o3d.visualization.draw_geometries([calibrated_pc])
        # draw_plotly([calibrated_pc])

def perception_node():
    perception = Perception()

    rospy.init_node('perception', anonymous=True)
    rospy.Subscriber("/zed2/zed_node/left/image_rect_color/compressed",
        CompressedImage, perception.callback_rgb,  queue_size = 1)
    rospy.Subscriber("/zed2/zed_node/point_cloud/cloud_registered",
        PointCloud2, perception.callback_pc,  queue_size = 10)
    
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    perception_node()
