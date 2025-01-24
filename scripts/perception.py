import numpy as np
import cv2
import rospy
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from open3d.visualization import draw_plotly
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.point_cloud2 import create_cloud
import std_msgs.msg
from matplotlib import colormaps

from ultralytics import YOLO

class Perception:
    def __init__(self):
        self.xyxy = None

    def calculate_angle(self, x1, y1, x2, y2, x3, y3, x4, y4):
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x4 - x3, y4 - y3])
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1, 1) 
        
        theta = np.arccos(cos_theta)
        theta = np.degrees(theta)
        
        return theta

    def cube_axes(self, x, y, z, yaw):
        translation = np.array([x, y, z])
        rotation_matrix = R.from_euler('z', yaw, degrees=True).as_matrix() 

        axis_length = 0.1

        # X-axis(red)
        x_axis = o3d.geometry.LineSet()
        x_axis.points = o3d.utility.Vector3dVector([
            translation, 
            translation + rotation_matrix[:, 0] * axis_length
        ])
        x_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
        x_axis.colors = o3d.utility.Vector3dVector([[1, 0, 0]]) 

        # Y-axis(green)
        y_axis = o3d.geometry.LineSet()
        y_axis.points = o3d.utility.Vector3dVector([
            translation, 
            translation + rotation_matrix[:, 1] * axis_length
        ])
        y_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
        y_axis.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

        # Z-axis(blue)
        z_axis = o3d.geometry.LineSet()
        z_axis.points = o3d.utility.Vector3dVector([
            translation, 
            translation + rotation_matrix[:, 2] * axis_length
        ])
        z_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
        z_axis.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  

        return x_axis, y_axis, z_axis


    def filter_pc(self, point_cloud_np, bboxes):
        if point_cloud_np.shape[1] == 4:
            point_cloud_np = point_cloud_np[:, :3]

        # K = [527.2972398956961, 0.0, 658.8206787109375, 0.0, 527.2972398956961, 372.25787353515625, 0.0, 0.0, 1.0]
        # fx, fy = K[0], K[4]
        # cx_cam, cy_cam = K[2], K[5]
        fx, fy = 527.2972398956961, 527.2972398956961
        cx_cam, cy_cam = 640, 360

        bboxes_ = bboxes.cpu().numpy()
        filtered_points = []
        labels = [] 
        label_stats = {} 
        for i, bbox in enumerate(bboxes_):  
            x1, y1, x2, y2 = bbox

            X, Y, Z = point_cloud_np[:, 0], point_cloud_np[:, 1], point_cloud_np[:, 2]
            u = (X * fx / Z) + cx_cam
            v = (Y * fy / Z) + cy_cam

            in_bbox = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            filtered_points.append(point_cloud_np[in_bbox])
            labels.append(np.full((np.sum(in_bbox), 1), i + 1))  # Assign bbox labels to points, starting from 1

        filtered_points = np.vstack(filtered_points)
        labels = np.vstack(labels)
        filtered_points = np.hstack((filtered_points, labels))  # Add labels as a fourth column

        # calibration
        extrinsic_rotation = R.from_quat([0.658734, 0.658652, 0.257135, 0.257155]).as_matrix()
        extrinsic_translation = np.array([0.209647, -0.0600195, 0.56205])
        points_base_frame = (extrinsic_rotation @ filtered_points[:, :3].T).T + extrinsic_translation
        valid_mask = points_base_frame[:, 2] > 0.0005  

        # Synchronously remove invalid points and labels
        points_base_frame = points_base_frame[valid_mask]
        labels = filtered_points[valid_mask, 3:4]  
        points_with_labels = np.hstack((points_base_frame, labels))
        
        # cube pose
        axes = [] 
        for label in np.unique(labels):
            points_for_label = points_with_labels[points_with_labels[:, 3] == label]
            
            xmax = points_for_label[:, 0].max()
            xmin = points_for_label[:, 0].min()
            ymax = points_for_label[:, 1].max()
            ymin = points_for_label[:, 1].min()
            zmax = points_for_label[:, 2].max()

            y_for_xmax = np.mean(points_for_label[points_for_label[:, 0] == xmax, 1]) 
            y_for_xmin = np.mean(points_for_label[points_for_label[:, 0] == xmin, 1])
            x_for_ymax = np.mean(points_for_label[points_for_label[:, 1] == ymax, 0])
            x_for_ymin = np.mean(points_for_label[points_for_label[:, 1] == ymin, 0])

            midpoint1_x = (xmax + xmin) / 2
            midpoint1_y = (y_for_xmax + y_for_xmin) / 2
            midpoint2_x = (x_for_ymax + x_for_ymin) / 2
            midpoint2_y = (ymax + ymin) / 2
            midpoint_x = (midpoint1_x + midpoint2_x) / 2
            midpoint_y = (midpoint1_y + midpoint2_y) / 2
            midpoint_z = zmax / 2

            yaw = self.calculate_angle(x1=xmin, y1=y_for_xmin, x2=x_for_ymax, y2=ymax, x3=0, y3=0, x4=1, y4=0)

            label_stats[label] = {
                "translation": (midpoint_x, midpoint_y, midpoint_z),
                "rotation": (0, 0, yaw)
            }
            
            # draw coordinate axes
            x_axis, y_axis, z_axis = self.cube_axes(midpoint_x, midpoint_y, midpoint_z, yaw)
            axes.extend([x_axis, y_axis, z_axis])

        print("Label Statistics:")
        for label, stats in label_stats.items():
            print(f"Cube {label}:")
            print(f"  translation: {stats['translation']}")
            print(f"  rotation: {stats['rotation']}")

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_with_labels[:, :3])
        
        # Assign a color to each point
        colors = colormaps["tab10"]
        point_colors = np.array([colors(int(label) % 10)[:3] for label in points_with_labels[:, 3]])  
        point_cloud.colors = o3d.utility.Vector3dVector(point_colors)



        o3d.visualization.draw_geometries([point_cloud] + axes)
        draw_plotly([point_cloud] + axes)
        # o3d.visualization.draw_geometries([point_cloud])
        # draw_plotly([point_cloud])

        return points_with_labels
    

    def callback_rgb(self, data):
        np_arr = np.frombuffer(data.data, np.uint8)
        rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
   
        model = YOLO("/opt/ros_ws/src/perception/model/best.pt")    # Load a trained model
        source = rgb_image
        results = model(source)                                     # return a list of Results objects

        for result in results:
            boxes = result.boxes                                    # Boxes object for bounding box outputs
            self.xyxy = boxes.xyxy
            result.save("/opt/ros_ws/src/perception/test_images/result0.jpg")

    def callback_pc(self, data):
        pc_data = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
        point_cloud_np = np.array(list(pc_data))

        if self.xyxy is not None:
            filtered_points_with_labels = self.filter_pc(point_cloud_np, self.xyxy)
            
            # Creating a PointCloud2 Message
            filtered_points_np = filtered_points_with_labels[:, :3]
            labels_np = filtered_points_with_labels[:, 3]
            
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "world"
            
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('label', 12, PointField.FLOAT32, 1),  
            ]
            
            # Combine points and labels
            combined_points = np.hstack((filtered_points_np, labels_np.reshape(-1, 1)))
            point_cloud_msg = create_cloud(header, fields, combined_points)
            
            pub.publish(point_cloud_msg)  

def perception():
    perception = Perception()

    rospy.init_node('perception', anonymous=True)
    rospy.Subscriber("/zed2/zed_node/left/image_rect_color/compressed",
        CompressedImage, perception.callback_rgb,  queue_size = 1)
    rospy.Subscriber("/zed2/zed_node/point_cloud/cloud_registered",
        PointCloud2, perception.callback_pc,  queue_size = 10)
    
    global pub
    pub = rospy.Publisher('filtered_point_cloud', PointCloud2, queue_size=10)
    
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    perception()