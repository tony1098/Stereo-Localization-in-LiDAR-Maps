# Stereo Localization in LiDAR Maps

Visual localization method in LiDAR maps. Only a stereo camera is need during localization since the LiDAR map can be built offline.

## 1. Prerequisites
1.1 ROS

1.2 Sophus (Lie algebra library)
```
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
```
## 2. Build 
The repository is a catkin package. To build, clone the repository and catkin_make:
```
cd ~/catkin_ws/src
git clone https://github.com/tony1098/Stereo-Localization.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```
## 3. Setup / Run
To run the package
```
rosrun stereo_localization stereo_localization_node
rosbag play YOUR_PATH_TO_DATASET/BAG_NAME.bag
```
The config file is located at res/config.yaml. Visualization information can be seen in rviz.

Also, since the localization is unable to run in real-time, the bag should **NOT** be played all in one go since it could result in overflowing the image buffer. That is, you may have to pause playing the bag so that the localization can catch up. 
## 4. Demo
The localization method can be tested on 2018-06-23-12-46-26_0.bag without any modifications to the config file. The corresponding LiDAR map is located at res/2018-06-23-12-46-26_0_z-filtered_4m.pcd.

**Video:**

<a href="https://www.youtube.com/embed/Pr21EpuHMjI" target="_blank"><img src="http://img.youtube.com/vi/Pr21EpuHMjI/0.jpg" 
alt="stereo_loc" width="240" height="180" border="10" /></a>
## 5. Future Work
5.1 The current localization depends on matching the stereo depth map to the LiDAR map. For more open scenes with faraway structures, good estimation of stereo depth is difficult. A possible solution is to use sparse but reliable 3D features computed from VIO directly for matching. This should speed up the localization runtime as well.

5.2 Currently, the localization can use the pose from VIO as an initial guess to further optimize the VIO pose using the LiDAR map. After optimization, the pose of VIO can be shifted directly according to the optimized transformation. One possible future work is to tightly-couple VIO and the LiDAR map based localization so that the entire VIO state vector is updated with each localization. 
