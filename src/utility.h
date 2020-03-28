#ifndef UTILITY_H
#define UTILITY_H

#include <fstream>
#include <string>
#include <vector>
#include <set>

// read bag
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH
#include <cv_bridge/cv_bridge.h>
#include "geometry_msgs/PoseStamped.h"

#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h> // pcl::transformPointCloud
#include <pcl/io/pcd_io.h> // read pcd file
#include <pcl/filters/extract_indices.h>
#include <pcl/octree/octree_search.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ximgproc.hpp> // for disparity map filtering
#include <opencv2/core/eigen.hpp> // for conversion between eigen

#include <sophus/se3.h>


extern Eigen::Quaterniond cam_pose_q;
extern Eigen::Vector3d cam_pose_t;
extern ros::Publisher  depth_map_pub, proj_img_pub, rect_left_img_pub, rect_right_img_pub, visible_submap_pub;
extern Eigen::Isometry3d LIDAR_TO_CAM_TRANSFORM, LIDAR_TO_VEHICLE_TRANSFORM;
extern int WIDTH, HEIGHT, X_ROI, Y_ROI, WIDTH_ROI, HEIGHT_ROI, NELDER_MEAD_ITERS;
extern double SCENE_SIZE;
extern cv::Mat K_l_new, depth_map;
extern pcl::PointCloud<pcl::PointXYZI>::Ptr submap, visible_submap;
extern int SKIPPED_IMAGES, LOCALIZATION_PER_FRAME;


void initializeGlobalParams();

bool inBorder(cv::Point2d projectedPt, const int MARGIN);

void computeDepthMap(const cv::Mat& left_img, const cv::Mat& right_img);

void storePCLMapInOctree();

void octreeRadiusSearch(float radius);

void projectSubmapToImage();

void extractVisibleSubmap(const int MARGIN);

double distance(const geometry_msgs::PoseStamped pre, const geometry_msgs::PoseStamped cur);

void build_map(const sensor_msgs::PointCloud2::ConstPtr& scan);

void mapping_get_lidar_poses();

#endif