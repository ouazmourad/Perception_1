import numpy as np
import cv2
import rospy
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from open3d.visualization import draw_plotly
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
from std_msgs.msg import Float64MultiArray
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.point_cloud2 import create_cloud
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
import std_msgs.msg
from matplotlib import colormaps
import tf.transformations as tft
from geometry_msgs.msg import Pose, PoseArray

from ultralytics import YOLO

class Perception:
    def __init__(self):
        self.xyxy = None
        self.rgb_image = None
        self.depth_image = None

    def knn(self, labels, points_with_labels):
        cube_centers = [] 
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

            cube_centers.append((midpoint_x, midpoint_y, midpoint_z))

        # reassign points by distance
        closest_labels = []
        for point_idx, point in enumerate(points_with_labels[:, :3]):
            distances = []
            for cube_idx, (cx, cy, cz) in enumerate(cube_centers):
                dist = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2 + (point[2] - cz)**2)
                distances.append((cube_idx + 1, dist))  # save (label, distance)

            # Sort by distance and select the closest label
            distances.sort(key=lambda x: x[1])
            closest_label = distances[0][0]
            points_with_labels[point_idx, 3] = closest_label  # Update point labels

            closest_labels.append(closest_label)
        
        return closest_labels, points_with_labels

    def knn_until_convergence(self, labels, points_with_labels):
        previous_labels = None
        iteration = 0

        while iteration < 50:
            closest_labels, points_with_labels = self.knn(labels, points_with_labels)

            if previous_labels is not None and np.array_equal(previous_labels, closest_labels):
                print(f"\nConverged after {iteration} iterations.\n")
                break

            previous_labels = closest_labels
            iteration += 1

        return closest_labels, points_with_labels

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

    def find_midpoint_z(self, pointcloud, closest_label):
        z_values = pointcloud[:, 2]
        bins_count = 100 # 50  

        counts, bins = np.histogram(z_values, bins=bins_count)

        max_bin_index = np.argmax(counts)
        max_bin_center = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2

        midpoint_z = max_bin_center / 2

        # Histogram
        plt.figure(figsize=(8, 5))
        plt.hist(z_values, bins=bins_count, color='blue', alpha=0.7, edgecolor='black')
        plt.xlabel("Z-Axis")
        plt.ylabel("Frequency")
        plt.title(f"Cube {closest_label} - Frequency Changes with Z-Axis of word-frame")
        plt.grid(True)

        plt.axvline(max_bin_center, color='red', linestyle='dashed', linewidth=2, label=f"Peak: {max_bin_center:.2f}")
        plt.legend()
        plt.savefig("/opt/ros_ws/src/perception/test_images/z_world_histogram")
        plt.close()
        # plt.show()

        return midpoint_z

    def find_midpoint_xy(self, pointcloud, midpoint_x, midpoint_y, yaw, closest_label):
        results_x = []
        results_y = []
        max_diff_x = -np.inf
        max_diff_y = -np.inf
        best_yaw_x = None
        best_yaw_y = None
        best_freq_x = None
        best_freq_y = None
        best_x = None
        best_y = None

        # x_values = pointcloud[:, 0]  # (before-world)
        # y_values = pointcloud[:, 1]

        for yaw in range(int(yaw) - 185, int(yaw) + 185, 1):  # 180 # 90 # 45
            cos_yaw, sin_yaw = np.cos(np.radians(-yaw)), np.sin(np.radians(-yaw)) 
            R_world_to_cube = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw,  cos_yaw]
            ])

            # convert X and Y axes of world-frame to cube-frame
            world_xy = pointcloud[:, :2]
            cube_xy = (R_world_to_cube @ (world_xy - np.array([midpoint_x, midpoint_y])).T).T  

            # Get the x and y value in the cube-frame
            cube_x_values = cube_xy[:, 0]  
            cube_y_values = cube_xy[:, 1] 

            # ———————————————————————————— X ————————————————————————————
            bins_count = 100 # 50 
            counts_x, bins_x = np.histogram(cube_x_values, bins=bins_count)

            bin_centers_x = (bins_x[:-1] + bins_x[1:]) / 2
            middle_index_x = len(bin_centers_x) // 2
            middle_x = bin_centers_x[middle_index_x]

            top3_indices_x = np.argsort(counts_x)[-3:][::-1] 
            top3_freqs_x = counts_x[top3_indices_x]  
            top3_x_vals = [(bins_x[i] + bins_x[i + 1]) / 2 for i in top3_indices_x]  

            # Calculate the difference between top_1_freq and top_2_freq
            diff_x = top3_freqs_x[0] - top3_freqs_x[1]
            if diff_x > max_diff_x:
                max_diff_x = diff_x
                best_yaw_x = yaw
                best_middle_x = middle_x
                best_freq_x = top3_freqs_x[0]
                best_x = top3_x_vals[0]

            results_x.append({
                "yaw": yaw,
                "top_1_freq": top3_freqs_x[0], "top_1_x": top3_x_vals[0],
                "top_2_freq": top3_freqs_x[1], "top_2_x": top3_x_vals[1],
                "top_3_freq": top3_freqs_x[2], "top_3_x": top3_x_vals[2]
            })

            # ———————————————————————————— Y ————————————————————————————
            counts_y, bins_y = np.histogram(cube_y_values, bins=bins_count)  

            bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2
            middle_index_y = len(bin_centers_y) // 2
            middle_y = bin_centers_y[middle_index_y]

            top3_indices_y = np.argsort(counts_y)[-3:][::-1]  
            top3_freqs_y = counts_y[top3_indices_y]  
            top3_y_vals = [(bins_y[i] + bins_y[i + 1]) / 2 for i in top3_indices_y]  

            diff_y = top3_freqs_y[0] - top3_freqs_y[1]  
            if diff_y > max_diff_y:
                max_diff_y = diff_y
                best_yaw_y = yaw
                best_middle_y = middle_y
                best_freq_y = top3_freqs_y[0]
                best_y = top3_y_vals[0]  

            results_y.append({
                "yaw": yaw,
                "top_1_freq": top3_freqs_y[0], "top_1_y": top3_y_vals[0],
                "top_2_freq": top3_freqs_y[1], "top_2_y": top3_y_vals[1],
                "top_3_freq": top3_freqs_y[2], "top_3_y": top3_y_vals[2]
            })

        # ———————————————————————————— Yaw based on the X-axis ————————————————————————————
        cos_yaw_x, sin_yaw_x = np.cos(np.radians(-best_yaw_x)), np.sin(np.radians(-best_yaw_x))
        R_world_to_cube_x = np.array([
            [cos_yaw_x, -sin_yaw_x],
            [sin_yaw_x,  cos_yaw_x]
        ])
        cube_xy_x = (R_world_to_cube_x @ (pointcloud[:, :2] - np.array([midpoint_x, midpoint_y])).T).T  
        best_cube_x_values = cube_xy_x[:, 0]
        
        # print("best_yaw_x:", best_yaw_x)
        # print("best_freq_x:", best_freq_x, "best_x:", best_x)
        # print("midpoint_x:", midpoint_x)

        ### Histogram ###
        plt.figure(figsize=(8, 5))
        # plt.hist(x_values, bins=bins_count, color='blue', alpha=0.7, edgecolor='black')            # (before-world)
        # plt.hist(cube_x_values, bins=bins_count, color='blue', alpha=0.7, edgecolor='black')       # before-cube
        plt.hist(best_cube_x_values, bins=bins_count, color='blue', alpha=0.7, edgecolor='black')    # after-cube
        plt.xlabel("X-Axis")
        plt.ylabel("Frequency")
        plt.title(f"Cube {closest_label} - Frequency Changes with X-Axis of cube-frame")
        plt.grid(True)
        plt.legend()
        plt.savefig("/opt/ros_ws/src/perception/test_images/x_cube_histogram")
        plt.close()
        # plt.show()
        
        ### Trend Chart ###
        yaws_x = [r["yaw"] for r in results_x]
        top_1_freqs_x = [r["top_1_freq"] for r in results_x]
        top_2_freqs_x = [r["top_2_freq"] for r in results_x]
        top_3_freqs_x = [r["top_3_freq"] for r in results_x]
        avg_top3_freqs_x = [(r["top_1_freq"] + r["top_2_freq"] + r["top_3_freq"]) / 3 for r in results_x]

        plt.figure(figsize=(10, 5))
        plt.plot(yaws_x, top_1_freqs_x, label="Top 1 Frequency (X)", marker='o')
        plt.plot(yaws_x, top_2_freqs_x, label="Top 2 Frequency (X)", marker='s')
        plt.plot(yaws_x, top_3_freqs_x, label="Top 3 Frequency (X)", marker='^')
        plt.plot(yaws_x, avg_top3_freqs_x, label="Top 3 Avg (X)", linewidth=3, linestyle='--', color='black')
        plt.xlabel("Yaw Angle (degrees)")
        plt.ylabel("Frequency")
        plt.title(f"Cube {closest_label} - Frequency Changes with Yaw Rotation (X Axis)")
        plt.legend()
        plt.grid(True)
        plt.savefig("/opt/ros_ws/src/perception/test_images/x_cube_trendchart")
        plt.close()
        # plt.show()

        # ———————————————————————————— Yaw based on the Y-axis ————————————————————————————
        cos_yaw_y, sin_yaw_y = np.cos(np.radians(-best_yaw_y)), np.sin(np.radians(-best_yaw_y))
        R_world_to_cube_y = np.array([
            [cos_yaw_y, -sin_yaw_y],
            [sin_yaw_y,  cos_yaw_y]
        ])
        cube_xy_y = (R_world_to_cube_y @ (pointcloud[:, :2] - np.array([midpoint_x, midpoint_y])).T).T  
        best_cube_y_values = cube_xy_y[:, 1]
        
        # print("best_yaw_y:", best_yaw_y)
        # print("best_freq_y:", best_freq_y, "best_y:", best_y)
        # print("midpoint_y:", midpoint_y)

        ### Histogram ###
        plt.figure(figsize=(8, 5))
        # plt.hist(y_values, bins=bins_count, color='blue', alpha=0.7, edgecolor='black')            # (before-world)
        # plt.hist(cube_y_values, bins=bins_count, color='blue', alpha=0.7, edgecolor='black')       # before-cube
        plt.hist(best_cube_y_values, bins=bins_count, color='blue', alpha=0.7, edgecolor='black')    # after-cube
        plt.xlabel("Y-Axis")
        plt.ylabel("Frequency")
        plt.title(f"Cube {closest_label} - Frequency Changes with Y-Axis of cube-frame")
        plt.grid(True)
        plt.legend()
        plt.savefig("/opt/ros_ws/src/perception/test_images/y_cube_histogram")
        plt.close()
        # plt.show()
        
        ### Trend Chart ###
        yaws_y = [r["yaw"] for r in results_y]
        top_1_freqs_y = [r["top_1_freq"] for r in results_y]
        top_2_freqs_y = [r["top_2_freq"] for r in results_y]
        top_3_freqs_y = [r["top_3_freq"] for r in results_y]
        avg_top3_freqs_y = [(r["top_1_freq"] + r["top_2_freq"] + r["top_3_freq"]) / 3 for r in results_y]

        plt.figure(figsize=(10, 5))
        plt.plot(yaws_y, top_1_freqs_y, label="Top 1 Frequency (Y)", marker='o')
        plt.plot(yaws_y, top_2_freqs_y, label="Top 2 Frequency (Y)", marker='s')
        plt.plot(yaws_y, top_3_freqs_y, label="Top 3 Frequency (Y)", marker='^')
        plt.plot(yaws_y, avg_top3_freqs_y, label="Top 3 Avg (Y)", linewidth=3, linestyle='--', color='black')
        plt.xlabel("Yaw Angle (degrees)")
        plt.ylabel("Frequency")
        plt.title(f"Cube {closest_label} - Frequency Changes with Yaw Rotation (Y Axis)")
        plt.legend()
        plt.grid(True)
        plt.savefig("/opt/ros_ws/src/perception/test_images/y_cube_trendchart")
        plt.close()
        # plt.show()

        #  ———————————————————————————— choose yaw based on X or Y ————————————————————————————
        best_yaw = best_yaw_x if max_diff_x > max_diff_y else best_yaw_y
        # print("max_diff_x", max_diff_x)
        # print("max_diff_y", max_diff_y)
        # print("best_yaw", best_yaw)

        cos_yaw, sin_yaw = np.cos(np.radians(best_yaw)), np.sin(np.radians(best_yaw))
        R_cube_to_world = np.array([
            [cos_yaw, sin_yaw], 
            [-sin_yaw, cos_yaw]
        ])
        best_middle_xy_world = R_cube_to_world @ np.array([best_middle_x, best_middle_y]) + np.array([midpoint_x, midpoint_y])
        midpoint_x = best_middle_xy_world[0]# + midpoint_x
        midpoint_y = best_middle_xy_world[1]# + midpoint_y
        # midpoint_x = cos_yaw * best_x - sin_yaw * 0 + midpoint_x
        # midpoint_y = sin_yaw * 0 + cos_yaw * best_y + midpoint_y

        # print("best_yaw:", best_yaw)
        # print("best_freq_x:", best_freq_x, "best_x:", best_x)
        # print("best_freq_y:", best_freq_y, "best_y:", best_y)
        # print("midpoint_x:", midpoint_x)
        # print("midpoint_y:", midpoint_y)

        return midpoint_x, midpoint_y, best_yaw

    # draw cube axes in Open3D(optional)
    def cube_axes(self, x, y, z, yaw):
        translation = np.array([x, y, z])
        rotation_matrix = R.from_euler('z', yaw, degrees=True).as_matrix() 

        axis_length = 0.05

        # --- X-axis(red) ---
        x_axis = o3d.geometry.LineSet()
        x_axis.points = o3d.utility.Vector3dVector([
            translation, 
            translation + rotation_matrix[:, 0] * axis_length
        ])
        x_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
        x_axis.colors = o3d.utility.Vector3dVector([[1, 0, 0]]) 

        # --- Y-axis(green) ---
        y_axis = o3d.geometry.LineSet()
        y_axis.points = o3d.utility.Vector3dVector([
            translation, 
            translation + rotation_matrix[:, 1] * axis_length
        ])
        y_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
        y_axis.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

        # --- Z-axis(blue) ---
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

        K = [527.2972398956961, 0.0, 658.8206787109375, 0.0, 527.2972398956961, 372.25787353515625, 0.0, 0.0, 1.0]
        # fx, fy = K[0], K[4]
        # cx_cam, cy_cam = K[2], K[5]
        fx, fy = 527.2972398956961, 527.2972398956961
        cx_cam, cy_cam = 640, 360
        # depth_image = depth_image / 2500.0   # maybe will be used in real world

        bboxes_ = bboxes.cpu().numpy()
        filtered_points = []
        labels = [] 
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

        points_base_frame = points_base_frame[valid_mask]
        labels = filtered_points[valid_mask, 3:4]  
        points_with_labels = np.hstack((points_base_frame, labels))
        
        ### cube pose ###
        closest_labels, points_with_labels = self.knn_until_convergence(labels, points_with_labels)
    
        # calculate accurate translation and rotation, draw coordinate axes
        label_stats = {} 
        axes = []
        for closest_label in np.unique(closest_labels):
            points_for_closest_label = points_with_labels[points_with_labels[:, 3] == closest_label]
            
            xmax = points_for_closest_label[:, 0].max()
            xmin = points_for_closest_label[:, 0].min()
            ymax = points_for_closest_label[:, 1].max()
            ymin = points_for_closest_label[:, 1].min()
            zmax = points_for_closest_label[:, 2].max()
            zmin = points_for_closest_label[:, 2].min()

            y_for_xmax = np.mean(points_for_closest_label[points_for_closest_label[:, 0] == xmax, 1]) 
            y_for_xmin = np.mean(points_for_closest_label[points_for_closest_label[:, 0] == xmin, 1])
            x_for_ymax = np.mean(points_for_closest_label[points_for_closest_label[:, 1] == ymax, 0])
            x_for_ymin = np.mean(points_for_closest_label[points_for_closest_label[:, 1] == ymin, 0])

            midpoint1_x = (xmax + xmin) / 2
            midpoint1_y = (y_for_xmax + y_for_xmin) / 2
            midpoint2_x = (x_for_ymax + x_for_ymin) / 2
            midpoint2_y = (ymax + ymin) / 2
            midpoint_x = (midpoint1_x + midpoint2_x) / 2
            midpoint_y = (midpoint1_y + midpoint2_y) / 2
            midpoint_z = (zmax + zmin) / 2

            yaw = self.calculate_angle(x1=xmin, y1=y_for_xmin, x2=x_for_ymax, y2=ymax, x3=0, y3=0, x4=1, y4=0)
            # print("midpoint_x:", midpoint_x)
            # print("midpoint_y:", midpoint_y)
            # print("midpoint_z:", midpoint_z)
            # print("yaw:", yaw)

            # midpoint_z = self.find_midpoint_z(points_for_closest_label, closest_label)
            # midpoint_x, midpoint_y, yaw = self.find_midpoint_xy(points_for_closest_label, midpoint_x, midpoint_y, yaw, closest_label)

            label_stats[closest_label] = {
                "translation": (midpoint_x, midpoint_y, midpoint_z),
                "rotation": (0, 0, yaw)
            }
            
            x_axis, y_axis, z_axis = self.cube_axes(midpoint_x, midpoint_y, midpoint_z, yaw)
            axes.extend([x_axis, y_axis, z_axis])

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
        # draw_plotly([point_cloud] + axes)

        return points_with_labels, label_stats

    def callback_rgb(self, data):
        np_arr = np.frombuffer(data.data, np.uint8)
        rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
   
        model = YOLO("/opt/ros_ws/src/perception/model/best.pt")    # Load a trained model
        source = rgb_image
        results = model(source)                                     # return a list of Results objects

        for result in results:
            boxes = result.boxes                                    # Boxes object for bounding box outputs
            self.xyxy = boxes.xyxy
            result.save("/opt/ros_ws/src/perception/test_images/detect_image.jpg")

    def callback_pc(self, data):
        # subscribe
        pc_data = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
        point_cloud_np = np.array(list(pc_data))

        # publish
        if self.xyxy is not None:
            filtered_points_with_labels, label_stats = self.filter_pc(point_cloud_np, self.xyxy)
            
            # create a PointCloud2 Message
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
            
            combined_points = np.hstack((filtered_points_np, labels_np.reshape(-1, 1)))
            point_cloud_msg = create_cloud(header, fields, combined_points)
            
            pub_pointcloud.publish(point_cloud_msg)  

            # create a PoseArray Message
            pose_array = PoseArray()
            pose_array.header.frame_id = "world"

            for label, stats in label_stats.items():
                pose = Pose()
                translation = stats['translation']
                rotation = stats['rotation']
                
                pose.position.x = translation[0]
                pose.position.y = translation[1]
                pose.position.z = translation[2]
                
                quaternion = R.from_euler('xyz', rotation, degrees=True).as_quat()
                pose.orientation.x = quaternion[0]
                pose.orientation.y = quaternion[1]
                pose.orientation.z = quaternion[2]
                pose.orientation.w = quaternion[3]
                

                
                pose_array.poses.append(pose)
            
            pub_cube_pose.publish(pose_array)

def perception():
    perception = Perception()

    rospy.init_node('perception', anonymous=True)
    rospy.Subscriber("/zed2/zed_node/left/image_rect_color/compressed", CompressedImage, perception.callback_rgb, queue_size = 1)
    # rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, perception.callback_pc, queue_size = 10)
    rospy.Subscriber("/zed2/zed_node/point_cloud/cloud_registered", PointCloud2, perception.callback_pc, queue_size = 10)
    
    global pub_pointcloud, pub_cube_pose
    pub_pointcloud = rospy.Publisher('filtered_point_cloud', PointCloud2, queue_size=10)
    pub_cube_pose = rospy.Publisher('cube_pose', PoseArray, queue_size=10)
    
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    perception()