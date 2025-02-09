import numpy as np
import cv2
import rospy
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.point_cloud2 import create_cloud
from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import Marker
import std_msgs.msg
import torch
from some_pointpillars_library import PointPillarsModel  # Replace with actual library

class Perception:
    def __init__(self):
        self.model = PointPillarsModel()  # Load the PointPillars model
        self.model.load_weights("/opt/ros_ws/src/perception/model/pointpillars.pth")  # Load trained model weights
        self.model.eval()

    def filter_pc(self, point_cloud_np):
        """
        Processes the raw point cloud and applies PointPillars for 3D object detection.
        """
        point_cloud_tensor = torch.tensor(point_cloud_np, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        detections = self.model(point_cloud_tensor)  # Run PointPillars inference
        
        objects = []
        for detection in detections:
            bbox = detection['bbox']  # 3D bounding box
            label = detection['label']
            confidence = detection['score']
            if confidence > 0.5:
                objects.append((bbox, label))

        return objects
    
    def callback_pc(self, data):
        """
        Processes incoming point cloud, runs PointPillars, and sends object poses.
        """
        pc_data = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
        point_cloud_np = np.array(list(pc_data))
        
        objects = self.filter_pc(point_cloud_np)
        
        pose_array = PoseArray()
        pose_array.header.frame_id = "world"

        for bbox, label in objects:
            pose = Pose()
            pose.position.x = bbox[0]
            pose.position.y = bbox[1]
            pose.position.z = bbox[2]
            
            rotation = R.from_euler('xyz', bbox[3:], degrees=True).as_quat()
            pose.orientation.x = rotation[0]
            pose.orientation.y = rotation[1]
            pose.orientation.z = rotation[2]
            pose.orientation.w = rotation[3]
            
            pose_array.poses.append(pose)

        pub_cube_pose.publish(pose_array)

    def grasp_planning(self, objects):
        """
        Uses detected 3D bounding boxes to plan grasp poses.
        """
        grasp_poses = []
        for bbox, label in objects:
            grasp_pose = Pose()
            grasp_pose.position.x = bbox[0]
            grasp_pose.position.y = bbox[1]
            grasp_pose.position.z = bbox[2] - 0.05  # Adjust for grasping height
            
            rotation = R.from_euler('xyz', [0, np.pi, 0], degrees=False).as_quat()
            grasp_pose.orientation.x = rotation[0]
            grasp_pose.orientation.y = rotation[1]
            grasp_pose.orientation.z = rotation[2]
            grasp_pose.orientation.w = rotation[3]
            
            grasp_poses.append(grasp_pose)
        
        return grasp_poses


def perception():
    rospy.init_node('perception', anonymous=True)
    rospy.Subscriber("/zed2/zed_node/point_cloud/cloud_registered", PointCloud2, Perception().callback_pc, queue_size=10)
    
    global pub_cube_pose
    pub_cube_pose = rospy.Publisher('cube_pose', PoseArray, queue_size=10)
    
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    perception()
