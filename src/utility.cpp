#include "utility.h"


static const float RESOLUTION = 128.0f;
static pcl::octree::OctreePointCloudSearch<pcl::PointXYZI> octree(RESOLUTION);
static pcl::PointCloud<pcl::PointXYZI>::Ptr map_pcl(new pcl::PointCloud<pcl::PointXYZI>() );

Eigen::Isometry3d LIDAR_TO_VEHICLE_TRANSFORM = Eigen::Isometry3d::Identity();
Eigen::Isometry3d LIDAR_TO_CAM_TRANSFORM = Eigen::Isometry3d::Identity();
static cv::Mat Q, JET_COLORMAP, M1_l, M2_l, M1_r, M2_r; // Q: disparity-to-depth mapping matrix
cv::Mat K_l_new;
static Eigen::Matrix3d eigen_K_l_new;
int WIDTH, HEIGHT, X_ROI, Y_ROI, WIDTH_ROI, HEIGHT_ROI, NELDER_MEAD_ITERS;

static cv::Mat rect_color_left_img, rect_color_right_img;
static int N_DISPARITIES, SAD_WINDOW_SIZE, PRE_FILTER_CAP;
double SCENE_SIZE;
cv::Mat depth_map;

static std::string LIDAR_MAP_FILEPATH;
int SKIPPED_IMAGES, LOCALIZATION_PER_FRAME;

// Mapping variables
static double MIN_SHIFT;
static int ADDED_NUM;
static std::map<double, geometry_msgs::PoseStamped> lidar_poses;
static std::string LIDAR_POSE_FILEPATH;


void initializeGlobalParams()
{
	// Load settings related to stereo calibration
	cv::FileStorage fsSettings("src/stereo_localization/res/config.yaml", cv::FileStorage::READ);
	if(!fsSettings.isOpened())
		ROS_INFO("ERROR: Wrong path to settings");

	cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r, LIDAR_TO_CAM_TRANS, LIDAR_TO_CAM_ROT;
	fsSettings["LEFT.K"] >> K_l;
	fsSettings["RIGHT.K"] >> K_r;
	fsSettings["LEFT.P"] >> P_l;
	fsSettings["RIGHT.P"] >> P_r;
	fsSettings["LEFT.R"] >> R_l;
	fsSettings["RIGHT.R"] >> R_r;
	fsSettings["LEFT.D"] >> D_l;
	fsSettings["RIGHT.D"] >> D_r;
	fsSettings["LIDAR_TO_CAM.rot"] >> LIDAR_TO_CAM_ROT;
	fsSettings["LIDAR_TO_CAM.trans"] >> LIDAR_TO_CAM_TRANS;
	fsSettings["Camera.width"] >> WIDTH;
	fsSettings["Camera.height"] >> HEIGHT;
	fsSettings["SGBM.n_disparities"] >> N_DISPARITIES;
	fsSettings["SGBM.sad_window_size"] >> SAD_WINDOW_SIZE;
	fsSettings["SGBM.pre_filter_cap"] >> PRE_FILTER_CAP;
	fsSettings["LIDAR_MAP_FILEPATH"] >> LIDAR_MAP_FILEPATH;
	fsSettings["SKIPPED_IMAGES"] >> SKIPPED_IMAGES;
	fsSettings["LOCALIZATION_PER_FRAME"] >> LOCALIZATION_PER_FRAME;
	fsSettings["SCENE_SIZE"] >> SCENE_SIZE;
	fsSettings["X_ROI"] >> X_ROI;
	fsSettings["Y_ROI"] >> Y_ROI;
	fsSettings["NELDER_MEAD_ITERS"] >> NELDER_MEAD_ITERS;
	fsSettings["MAPPING.min_shift"] >> MIN_SHIFT;
	fsSettings["MAPPING.added_num"] >> ADDED_NUM;
	fsSettings["MAPPING.lidar_pose_filepath"] >> LIDAR_POSE_FILEPATH;

	if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() )
		ROS_INFO("ERROR: Calibration parameters to rectify stereo are missing!");

	Eigen::Matrix3d rot;
    Eigen::Vector3d trans;
	for(int row = 0; row < 3; row++)
	{
		for(int col = 0; col < 3; col++)
		{
			rot(row, col) = LIDAR_TO_CAM_ROT.at<double>(row, col);
		}
	}
	for(int row = 0; row < 3; row++)
	{
		trans(row) = LIDAR_TO_CAM_TRANS.at<double>(row);
	}
	
    LIDAR_TO_CAM_TRANSFORM.rotate(rot);
    LIDAR_TO_CAM_TRANSFORM.pretranslate(trans);
	
	Eigen::Isometry3d T_vehicle_pose = Eigen::Isometry3d::Identity();
	// initial vehicle pose: set to map origin
	Eigen::Quaterniond vehicle_pose_q(1.0, 0.0, 0.0, 0.0); // w, x, y, z
	Eigen::Vector3d vehicle_pose_t(0.095349, 0.707931, -0.002151);

	T_vehicle_pose.rotate(vehicle_pose_q);
	T_vehicle_pose.pretranslate(vehicle_pose_t); 

	LIDAR_TO_VEHICLE_TRANSFORM = Eigen::Isometry3d::Identity();
	Eigen::AngleAxisd rot_v(-2.1, Eigen::Vector3d(0, 0, 1));
	LIDAR_TO_VEHICLE_TRANSFORM.rotate(rot_v);
	LIDAR_TO_VEHICLE_TRANSFORM.pretranslate(Eigen::Vector3d(1.2, 0, 2));

	// Set initial camera pose
	Eigen::Isometry3d T_lidar_pose = T_vehicle_pose * LIDAR_TO_VEHICLE_TRANSFORM;
	Eigen::Isometry3d T_cam_pose = T_lidar_pose * LIDAR_TO_CAM_TRANSFORM.inverse(); 
	cam_pose_q = T_cam_pose.rotation();
	cam_pose_t = T_cam_pose.translation();

	double r[3][3] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
	double t[3] = {-0.12, 0.0, 0.0}; // baseline
	cv::Mat R (3, 3, CV_64FC1, r);
	cv::Mat T (3, 1, CV_64FC1, t), R1, R2, P1, P2;

	WIDTH_ROI = WIDTH - X_ROI, HEIGHT_ROI = HEIGHT - 100;
	cv::Rect validPixROI1, validPixROI2;	
	cv::stereoRectify(K_l, D_l, K_r, D_r, cv::Size(WIDTH, HEIGHT), R, T, R1, R2, P1, P2, Q, 
						CV_CALIB_ZERO_DISPARITY, 1, cv::Size(WIDTH, HEIGHT), &validPixROI1, &validPixROI2);

	K_l_new = P_l.rowRange(0,3).colRange(0,3);
	for(int row = 0; row < 3; row++)
	{
		for(int col = 0; col < 3; col++)
		{
			eigen_K_l_new(row, col) = K_l_new.at<double>(row, col);
		}
	}
	cv::initUndistortRectifyMap(K_l, D_l, R_l, K_l_new,
										cv::Size(WIDTH, HEIGHT), CV_32FC1, M1_l, M2_l);
	cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0,3).colRange(0,3),
										cv::Size(WIDTH, HEIGHT), CV_32FC1, M1_r, M2_r);
	JET_COLORMAP = cv::imread("src/stereo_localization/res/colorscale_jet.jpg", CV_LOAD_IMAGE_UNCHANGED);
}

void computeDepthMap(const cv::Mat& left_img, const cv::Mat& right_img)
{
	cv::Mat imgLeft, imgRight;
	// convert to grayscale
	cv::Mat left_gray_img, right_gray_img;		
	cv::cvtColor(left_img, left_gray_img, cv::COLOR_RGB2GRAY);		
	cv::cvtColor(right_img, right_gray_img, cv::COLOR_RGB2GRAY);
	// rectify
	cv::remap(left_gray_img, imgLeft, M1_l, M2_l, cv::INTER_LINEAR);
	cv::remap(right_gray_img, imgRight, M1_r, M2_r, cv::INTER_LINEAR);
	cv::remap(left_img, rect_color_left_img, M1_l, M2_l, cv::INTER_LINEAR); // needed for projectSubmapToImage()
	cv::remap(right_img, rect_color_right_img, M1_r, M2_r, cv::INTER_LINEAR);

	//SGBM
	cv::Mat left_disp, right_disp;
	int mindisparity = 0;
	cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(mindisparity, N_DISPARITIES, SAD_WINDOW_SIZE);
	const int P1 = 8 * imgLeft.channels() * SAD_WINDOW_SIZE * SAD_WINDOW_SIZE;
	const int P2 = 32 * imgRight.channels() * SAD_WINDOW_SIZE * SAD_WINDOW_SIZE;
	left_matcher->setP1(P1);
	left_matcher->setP2(P2);
	left_matcher->setPreFilterCap(PRE_FILTER_CAP);
	left_matcher->compute(imgLeft, imgRight, left_disp);

	// disparity map filtering
	cv::Mat filtered_disp;
	cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
	cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);	
	right_matcher->compute(imgRight, imgLeft, right_disp);
	double lambda = 8000.0; 
	double sigma = 1.5;
	wls_filter->setLambda(lambda);
	wls_filter->setSigmaColor(sigma);
	wls_filter->filter(left_disp, rect_color_left_img, filtered_disp, right_disp);
    // Divide by 16 to get true disparity values
	filtered_disp.convertTo(filtered_disp, CV_32F, 1.0 / 16); 

	cv::reprojectImageTo3D(filtered_disp, depth_map, Q, true, -1); 

	// all depth < 0 or > SCENE_SIZE and depth == inf (disp == 0) is set to SCENE_SIZE
	for(int y = 0; y < depth_map.rows; y++)
    {
        for(int x = 0; x < depth_map.cols; x++)
        {
			if(std::isinf(depth_map.at<cv::Vec3f>(y, x).val[2]) || 
					  depth_map.at<cv::Vec3f>(y, x).val[2] <= 0 ||
					  depth_map.at<cv::Vec3f>(y, x).val[2] > SCENE_SIZE)
            {
				depth_map.at<cv::Vec3f>(y, x).val[2] = SCENE_SIZE;
            }
        }
    }

	// split depth map along z axis
	cv::Mat xyz[3];
	cv::split(depth_map, xyz);
	cv::convertScaleAbs( xyz[2], xyz[2], 255.0 / SCENE_SIZE );
	cv::applyColorMap( xyz[2], xyz[2], cv::COLORMAP_JET );
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", xyz[2]).toImageMsg();
	depth_map_pub.publish(*msg);
}

void storePCLMapInOctree()
{
	if (pcl::io::loadPCDFile<pcl::PointXYZI> (LIDAR_MAP_FILEPATH, *map_pcl) == -1) //* load the file
		ROS_INFO("Couldn't read .pcd file!");
	
	std::cout << "Loaded "
				<< map_pcl->width * map_pcl->height
				<< " data points from .pcd"
				<< std::endl;

	octree.setInputCloud(map_pcl);
	octree.addPointsFromInputCloud();
}

// returned submap in MAP coords
void octreeRadiusSearch(float radius)
{
	pcl::PointXYZI searchPoint;
	searchPoint.x = cam_pose_t(0);
	searchPoint.y = cam_pose_t(1);
	searchPoint.z = cam_pose_t(2);

  	std::vector<float> pointRadiusSquaredDistance;
	std::vector<int> pointIdxRadiusSearch;
	
	if (octree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	{
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
		pcl::ExtractIndices<pcl::PointXYZI> extract;
		inliers->indices = pointIdxRadiusSearch;
		extract.setInputCloud(map_pcl);
		extract.setIndices(inliers);
		extract.setNegative(false);
		extract.filter(*submap);
	}
	else
		ROS_INFO("octree radius search <= 0");
}

bool comparePoint(const pcl::PointXYZI& p1, const pcl::PointXYZI& p2)
{
	if(p1.z < p2.z)
		return true;
	else
		return false;
}

void projectSubmapToImage()
{
    cv::Mat proj_img = rect_color_left_img.clone();
    cv::Mat rect_right_img = rect_color_right_img.clone();
	sensor_msgs::ImagePtr left_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", proj_img).toImageMsg();
	sensor_msgs::ImagePtr right_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rect_right_img).toImageMsg();
	rect_left_img_pub.publish(*left_msg);
	rect_right_img_pub.publish(*right_msg);
	
	std::vector<cv::Point3d> points_3d; // in camera coordinates
	std::vector<double> points3d_depth_mapping;
	for(int i = 0; i < visible_submap->size(); i++) 
	{
		double x = visible_submap->at(i).x;
		double y = visible_submap->at(i).y;
		double z = visible_submap->at(i).z;

		if(z > 0)
		{
			points_3d.push_back(cv::Point3d(x, y, z) );
			points3d_depth_mapping.push_back(z);
		}
	}

    std::vector<cv::Point2d> projectedPoints; // only contains projections from 3D point where z > 0
	cv::Mat empty;
    cv::Mat rVec(3, 1, cv::DataType<double>::type); 
    cv::Mat tVec(3, 1, cv::DataType<double>::type); 
    rVec.at<double>(0) = 0;
	rVec.at<double>(1) = 0;
	rVec.at<double>(2) = 0;
	tVec.at<double>(0) = 0;
	tVec.at<double>(1) = 0;
	tVec.at<double>(2) = 0;
	projectPoints(points_3d, rVec, tVec, K_l_new, empty, projectedPoints);

	for(int i = projectedPoints.size() - 1; i >= 0 ; i--)
	{
		cv::Point2d projectedPt = projectedPoints[i];

		int round_x = round(projectedPt.x);
		int round_y = round(projectedPt.y);

		// apply colormap
		double scaled_depth = fabs(points3d_depth_mapping[i] * 255.0 / SCENE_SIZE );
		if(scaled_depth > 255)
			scaled_depth = 255;
		cv::Vec3b intensity = JET_COLORMAP.at<cv::Vec3b>(0, scaled_depth);

		cv::circle(proj_img, cv::Point(round(projectedPt.x), round(projectedPt.y) ), 2, 
								cv::Scalar(intensity[0], intensity[1], intensity[2]), -1); // visible_submap projected points
	}
	/*
	// draw bounding rectangle
	validPixROI1 = cv::Rect(X_ROI, Y_ROI, WIDTH_ROI, HEIGHT_ROI);
	cv::rectangle(proj_img, validPixROI1, cv::Scalar(255, 255, 255), 2);
	*/
    left_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", proj_img).toImageMsg();
	proj_img_pub.publish(*left_msg);
}

// returned visible_submap is in CAMERA coordinates
void extractVisibleSubmap(const int MARGIN)
{
	pcl::copyPointCloud(*submap, *visible_submap);
	int orig_num_points = visible_submap->points.size();

	// update lidar poses using extrinsics
	Eigen::Isometry3d T_cam_pose = Eigen::Isometry3d::Identity();
	T_cam_pose.rotate(cam_pose_q);
	T_cam_pose.pretranslate(cam_pose_t); 

	// map to camera coordinates	
	pcl::transformPointCloud (*visible_submap, *visible_submap, T_cam_pose.inverse().matrix() );
	std::sort(visible_submap->points.begin(), visible_submap->points.end(), comparePoint);
	
	pcl::PointIndices::Ptr outliers(new pcl::PointIndices()); 
	std::vector<cv::Point3d> points_3d; // in camera coordinates
    std::vector<int> points3d_visible_submap_mapping;
	for(int i = 0; i < visible_submap->size(); i++) 
	{
		double x = visible_submap->at(i).x;
		double y = visible_submap->at(i).y;
		double z = visible_submap->at(i).z;
		if(z > 0)
        {
			points_3d.push_back(cv::Point3d(x, y, z) );
            points3d_visible_submap_mapping.push_back(i);
        }
		else
			outliers->indices.push_back(i);
	}

    std::vector<cv::Point2d> projectedPoints; // only contains projections from 3D point where z > 0
	cv::Mat empty;
    cv::Mat rVec(3, 1, cv::DataType<double>::type); 
    cv::Mat tVec(3, 1, cv::DataType<double>::type); 
    rVec.at<double>(0) = 0;
	rVec.at<double>(1) = 0;
	rVec.at<double>(2) = 0;
	tVec.at<double>(0) = 0;
	tVec.at<double>(1) = 0;
	tVec.at<double>(2) = 0;
	projectPoints(points_3d, rVec, tVec, K_l_new, empty, projectedPoints);
	cv::Mat depth_buffer(HEIGHT, WIDTH, CV_32FC1, cv::Scalar(100)); // for occlusion filter
	pcl::ExtractIndices<pcl::PointXYZI> extract;

	int occluded_count = 0;
	for(int i = 0; i < projectedPoints.size(); i++)
	{
		cv::Point2d projectedPt = projectedPoints[i];

		int index = points3d_visible_submap_mapping[i];
		if(inBorder(projectedPt, MARGIN)) // project inside image, but may be occluded
		{
			
			int round_x = round(projectedPt.x);
			int round_y = round(projectedPt.y);
			// occlusion filter
			if(depth_buffer.at<float>(round_y, round_x) > points_3d[i].z)
			{
				for(int y_shift = -1; y_shift <= 1; y_shift++) // 3x3 patch
				{
					for(int x_shift = -1; x_shift <= 1; x_shift++)
					{
						depth_buffer.at<float>(round_y + y_shift, round_x + x_shift) = points_3d[i].z;			
					}
				}			
			}
			else // occluded
			{
           		outliers->indices.push_back(index);
				occluded_count++;
			}
		}
		else 
            outliers->indices.push_back(index); // keep only lidar points that are within camera viewing frustum
	}
	std::cout << "#submap points: " << orig_num_points << " -> " << orig_num_points - occluded_count << std::endl;

	extract.setInputCloud(visible_submap);
	extract.setIndices(outliers);
	extract.setNegative(true);
	extract.filter(*visible_submap);
	
	// return submap to lidar coords
	pcl::PointCloud<pcl::PointXYZI>::Ptr visible_submap_lidar_coords(new pcl::PointCloud<pcl::PointXYZI>() ); 
	pcl::transformPointCloud(*visible_submap, *visible_submap_lidar_coords, LIDAR_TO_CAM_TRANSFORM.inverse().matrix() );
	// publish visible_submap
	sensor_msgs::PointCloud2 visible_submap_ros;
	pcl::toROSMsg(*visible_submap_lidar_coords, visible_submap_ros);
	visible_submap_ros.header.frame_id = "map";
	visible_submap_pub.publish(visible_submap_ros);
}

bool inBorder(cv::Point2d projectedPt, const int MARGIN)
{
	if(projectedPt.x > (X_ROI + WIDTH_ROI - 1) - MARGIN || projectedPt.x < X_ROI + MARGIN || 
					projectedPt.y > (Y_ROI + HEIGHT_ROI - 1) - MARGIN || projectedPt.y < Y_ROI + MARGIN)
		return false;
	else
		return true;
}

double distance(const geometry_msgs::PoseStamped pre, const geometry_msgs::PoseStamped cur)
{
	return std::sqrt(std::pow(cur.pose.position.x-pre.pose.position.x,2)+std::pow(cur.pose.position.y-pre.pose.position.y,2)+std::pow(cur.pose.position.z-pre.pose.position.z,2));
}

void build_map(const sensor_msgs::PointCloud2::ConstPtr& scan)
{
	static bool written = false;
	static int added_num = 0;
	static geometry_msgs::PoseStamped last_added_pose;
	static bool isMapInitial = true;
	static pcl::PointCloud<pcl::PointXYZI> map;

	// write to map
	if (added_num == ADDED_NUM && !written)
	{
		std::string pcd_filename = "map.pcd";
		if(pcl::io::savePCDFileBinary(pcd_filename, map) == -1)
			std::cout << "Failed saving " << pcd_filename << "." << "\n";
		std::cout << "Saved " << pcd_filename << " (" << map.size() << " points)" << "\n";
		written = true;
	}
	
	// make sure we have lidar pose for current scan
	if(lidar_poses.find(scan->header.stamp.toSec()) == lidar_poses.end() )
		return;

	geometry_msgs::PoseStamped cur_scan_pose = lidar_poses[scan->header.stamp.toSec()];		
	

	// filter some scans for mapping
	if (!isMapInitial && distance(last_added_pose, cur_scan_pose) < MIN_SHIFT)
	{
		ROS_INFO("Shift: %f", distance(last_added_pose, cur_scan_pose) );
		return;
	}
	
	// convert to pcl format
	pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_scan(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::fromROSMsg(*scan, *pcl_scan);

	// filter points on car
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices()); 
	pcl::ExtractIndices<pcl::PointXYZI> extract;
	for(int i = 0; i < pcl_scan->points.size(); i++)
	{
		double x = pcl_scan->points[i].x;
		double y = pcl_scan->points[i].y;
		double z = pcl_scan->points[i].z;

		if(x*x + y*y + z*z > 3)
            inliers->indices.push_back(i); // keep only lidar points that are within sqrt(3) of origin
	}

	extract.setInputCloud(pcl_scan);
	extract.setIndices(inliers);
	extract.setNegative(false);
	extract.filter(*pcl_scan);
	
	// transform to base_link coordinates
	Eigen::Isometry3d lidar_to_vehicle = Eigen::Isometry3d::Identity();
	Eigen::AngleAxisd rot_v(-2.1, Eigen::Vector3d(0, 0, 1));
	lidar_to_vehicle.rotate(rot_v);
	lidar_to_vehicle.pretranslate(Eigen::Vector3d(1.2, 0, 2));
	pcl::transformPointCloud (*pcl_scan, *pcl_scan, lidar_to_vehicle.matrix() );

	Eigen::Matrix3f rot_mat = Eigen::Quaternionf(cur_scan_pose.pose.orientation.w, cur_scan_pose.pose.orientation.x, cur_scan_pose.pose.orientation.y, cur_scan_pose.pose.orientation.z).toRotationMatrix(); 
	Eigen::Matrix4f tf_btow = Eigen::Matrix4f::Identity();
	tf_btow.block(0,0,3,3) = rot_mat;
	tf_btow(0,3) = cur_scan_pose.pose.position.x;
	tf_btow(1,3) = cur_scan_pose.pose.position.y;
	tf_btow(2,3) = cur_scan_pose.pose.position.z;
	pcl::transformPointCloud (*pcl_scan, *pcl_scan, tf_btow);

	if (isMapInitial)
	{
		map = *pcl_scan;
		isMapInitial = false;
		added_num = 1;
	}
	else
	{
		map += *pcl_scan;
		added_num++;
	}
	last_added_pose = cur_scan_pose;
	ROS_INFO("Added %d scans in total", added_num);
}

void mapping_get_lidar_poses()
{
	// go thru lidar data to get poses to build map
	rosbag::Bag bag;
	bag.open(LIDAR_POSE_FILEPATH, rosbag::bagmode::Read);
	// bag.open("/home/tony/Documents/rosbags/poses/0326_pose.bag", rosbag::bagmode::Read);
	std::vector<std::string> topics;
    topics.push_back(std::string("/current_pose"));
	rosbag::View view(bag, rosbag::TopicQuery(topics));
	foreach(rosbag::MessageInstance const m, view)
    {
        geometry_msgs::PoseStamped::Ptr pose = m.instantiate<geometry_msgs::PoseStamped>();
		if (pose != NULL)
		{
			pose->header.frame_id = "map";
			lidar_poses[pose->header.stamp.toSec()] = *pose;
		}
    }
	bag.close();
	ROS_INFO("Finished reading lidar poses!");
}