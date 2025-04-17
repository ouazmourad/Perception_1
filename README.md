# 🦾 IROBMAN Final Project – Cube Pose Estimation & Manipulation

## 📌 Objective
This project aims to autonomously detect, estimate, and manipulate colored cubes on a table using a Franka Emika Panda robot. The robot builds a cube tower using vision-based detection and custom control planning. Unlike prior implementations that relied entirely on standard pipelines (like MoveIt Pick&Place), our approach integrates custom perception, pose refinement, and filtering for robust cube handling.

---

## 🧠 System Overview

### 🔍 Perception

Our perception pipeline is designed for robust, real-time cube detection and 6D pose estimation.

- **YOLOv8-based Object Detection**: Fine-tuned YOLO model to detect 2D bounding boxes of cubes from RGB images.
- **3D Pointcloud Filtering**: Using aligned depth data, we filter points inside YOLO bounding boxes and above table height (Z > 0).
- **Pose Refinement**:
  - **KNN clustering** separates cubes in close proximity.
  - **Histogram analysis** (X/Y axes) to extract dominant surface coordinates.
  - **PCA** is applied to estimate yaw orientation.
- **Cube-to-World Transformation**: Real-time TF transforms applied to convert cube pose from camera frame to world frame.
- **Offset Correction**: 2 cm offset applied to shift cube center away from wall-aligned estimates.

### 📦 Perception Output
- `PoseArray` of detected cube poses.
- `PointCloud2` of filtered, labeled points.

---

### 🤖 Control

The control module manages precise manipulation:

- **Custom Grasp Planner**: A custom logic pipeline replaces the default MoveIt grasp planner.
- **Gripper Control**: Direct control via `franka_gripper` package (velocity and force).
- **Self-Correcting Behavior**: Retries failed grasps or adjusts grasp position.
- **Trajectory Planning**: Uses MoveIt! with CHOMP and collision objects.

---

### 🧠 Planning

The planning system drives sequential execution and pose matching:

- **State Machine**: Built-in ROS state machine for detect → pick → place cycles.
- **Cube Hypothesis Tracking**:
  - Compares incoming poses with previously stored ones using Euclidean distance.
  - Distinguishes between existing cubes and new detections.
- **Tower Planning**: Maintains a stack counter to increment placement height per new cube.

---

## 🧪 Key Features

- 📸 **Supports Top-down & Side Views**  
- 📈 **Histograms & Trend Charts for Pose Refinement**  
- 🔁 **Pose Averaging for Stability**  
- 🧠 **Simple Pose Matching for Consistent Cube Tracking**  
- 🧱 **Stacking Logic to Build a Cube Tower**

---

## 🧑‍💻 Contributions

| Name     | Contributions                                                                 |
|----------|--------------------------------------------------------------------------------|
| **Yuan**   | Pose estimation, Open3D visualizations, dataset & training                  |
| **Mourad** | YOLO integration, PCA & KNN pose refinement, filtering logic                |
| **Dustin** | Initial ROS pipeline, system integration, robot setup                       |
| **Tim**    | Grasp planning, tower stacking logic, gripper coordination                  |

---

## 📹 Demo Link

https://youtu.be/_FC3Sgik9uQ?si=fv6sOGOaYNJWwfS_

---

## 🚧 Future Improvements

- Robust tracking across time with Kalman filtering.
- Fall detection: dropped cubes or tower collapse.
- Improved cube ID consistency across frames.
- GUI for visual status & debug info.

---

## 🛠️ Tech Stack

- ROS Noetic  
- YOLOv8 (Ultralytics)  
- Open3D  
- OpenCV & matplotlib  
- Franka Emika Panda + Gripper  
- MoveIt! + CHOMP Planner  

---

## 📁 Repo Structure (Recommended)

