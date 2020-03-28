#include <stdio.h>
#include <cmath>
#include <utility> // std::pair
#include <queue>
#include <thread>
#include <mutex>
#include <map>
#include <fstream>

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/PointCloud2.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utility.h"
#include "nelder_mead.h"


static std::mutex r_buf, l_buf;
static std::queue<sensor_msgs::Image> left_img_buf, right_img_buf;
ros::Publisher depth_map_pub, proj_img_pub, rect_left_img_pub, rect_right_img_pub, visible_submap_pub;
pcl::PointCloud<pcl::PointXYZI>::Ptr submap(new pcl::PointCloud<pcl::PointXYZI>() ); // in MAP coords: submap containing points <= radius
pcl::PointCloud<pcl::PointXYZI>::Ptr visible_submap(new pcl::PointCloud<pcl::PointXYZI>() ); // in CAMERA coords

// rotation quaternion (x, y, z, w) + translation(x, y, z)
Eigen::Quaterniond cam_pose_q;
Eigen::Vector3d cam_pose_t;


void left_img_callback(const sensor_msgs::Image::ConstPtr& img)
{	
	l_buf.lock();
	left_img_buf.push(*img);
	l_buf.unlock();
}

void right_img_callback(const sensor_msgs::Image::ConstPtr& img)
{
	r_buf.lock();
	right_img_buf.push(*img);
	r_buf.unlock();
}

void process()
{
	static bool isInitial = true;
	storePCLMapInOctree();
	
	Nelder_Mead Solver;

	while(true)
	{
		std::vector<std::pair<sensor_msgs::Image, sensor_msgs::Image> > measurements;
		measurements.clear();

		l_buf.lock();
		r_buf.lock();
		while(true)
		{
			if( left_img_buf.empty() || right_img_buf.empty() )
				break;
			int num_measurements = ( (left_img_buf.size() <= right_img_buf.size()) ? 
										left_img_buf.size() : right_img_buf.size() );		
			while(num_measurements--)
			{
				measurements.push_back(std::make_pair(left_img_buf.front(), right_img_buf.front() ));
				left_img_buf.pop();
				right_img_buf.pop();
			}
		}
		l_buf.unlock();
		r_buf.unlock();

		if(measurements.size() == 0) // go back if empty
			continue;		

		static int img_num = 0;
		for(auto &measurement : measurements)
		{
			std::cout << "IMG #"  << img_num << std::endl;

			if(img_num > SKIPPED_IMAGES)
			{	
				octreeRadiusSearch(40);

				// convert to Mat
				cv_bridge::CvImagePtr cv_left_img = cv_bridge::toCvCopy(measurement.first, "8UC3");
				cv::Mat left_img = cv_left_img->image;
				cv_bridge::CvImagePtr cv_right_img = cv_bridge::toCvCopy(measurement.second, "8UC3");
				cv::Mat right_img = cv_right_img->image;

				computeDepthMap(left_img, right_img);			
				
				if(img_num % LOCALIZATION_PER_FRAME == 0) 
				{				
					extractVisibleSubmap(60); 

					if(isInitial)
					{
						isInitial = false;
						projectSubmapToImage();	
					}
					/*
					// set variance
					std::vector<double> variance;
					variance.reserve(visible_submap->points.size());
					for(int i = 0; i < visible_submap->points.size(); i++)
					{
						double val = visible_submap->at(i).z; 
						
						if(val <= 10)
							val = 1.0;
						else 
							val = 1 + 0.05 * (val - 10);
						
						variance.push_back(value);
					}
					setVariance(variance);
					*/
					// DO OPTIMIZATION 
					Sophus::Vector6d se3 = Sophus::Vector6d::Zero();
					Solver.solve(se3.data(), 6); // Downhill simplex
					std::cout << "After optimization: " << se3 << std::endl;

					// UPDATE CURRENT POSE
					Eigen::Map<const Eigen::Matrix<double, 6, 1> > T_se3(se3.data() ); // update amount
					Sophus::SE3 T_SE3 = Sophus::SE3::exp(T_se3); // T_SE3 is update transformation
					Eigen::Isometry3d T_cam_shift = Eigen::Isometry3d::Identity();		
					T_cam_shift.rotate( T_SE3.rotation_matrix() );
					T_cam_shift.pretranslate( T_SE3.translation() );
					Eigen::Isometry3d T_shift_cam = T_cam_shift.inverse();
					
					Eigen::Isometry3d T_shift_pose = Eigen::Isometry3d::Identity();
					T_shift_pose.rotate(cam_pose_q);
					T_shift_pose.pretranslate(cam_pose_t);

					T_shift_pose = T_shift_pose * T_shift_cam;
					cam_pose_q = T_shift_pose.rotation();
					cam_pose_t = T_shift_pose.translation();
					// update visible_submap transformation for projectSubmapToImage()
					pcl::transformPointCloud (*visible_submap, *visible_submap, T_cam_shift.matrix() );
					
					projectSubmapToImage();
				}
			}
			img_num++;
		}
	}
}

int main(int argc, char **argv)
{
  	ros::init(argc, argv, "stereo_localization");
  	ros::NodeHandle n;

	initializeGlobalParams();

	// create publishers
	depth_map_pub = n.advertise<sensor_msgs::Image>("depth_map", 1000);
	proj_img_pub = n.advertise<sensor_msgs::Image>("projected_image", 1000);
	rect_left_img_pub = n.advertise<sensor_msgs::Image>("rect_left_img", 1000);
	rect_right_img_pub = n.advertise<sensor_msgs::Image>("rect_right_img", 1000);
	visible_submap_pub = n.advertise<sensor_msgs::PointCloud2>("visible_submap", 1000);

	// subscribers
	// mapping_get_lidar_poses();
	// ros::Subscriber sub_lidar = n.subscribe("/points_raw", 2000, build_map); 
	image_transport::ImageTransport it(n);
	image_transport::TransportHints th("compressed");
  	image_transport::Subscriber sub_left_cam = it.subscribe("/zed/left/image_raw_color", 2000, left_img_callback, th);
	image_transport::Subscriber sub_right_cam = it.subscribe("/zed/right/image_raw_color", 2000, right_img_callback, th);

	std::thread measurement_process{process};

	ros::spin();
  	return 0;
}

