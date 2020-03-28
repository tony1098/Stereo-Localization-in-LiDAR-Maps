#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#include "utility.h"


std::vector<double> g_variance;

struct NMpoint{
    Eigen::VectorXd vec;
    double score;
};

bool inBorder(cv::Point2d projectedPt)
{
	if(projectedPt.x > WIDTH - 1 || projectedPt.x < X_ROI || 
					projectedPt.y > HEIGHT - 1 || projectedPt.y < 0)
		return false;
	else
		return true;
}

double huber_norm(double x, double alpha)
{
    if(fabs(x) < alpha)
        return x * x;
    else
        return 2 * (fabs(x) - alpha) + alpha*alpha; // modified to slope = 2
        // original huber norm: 2*alpha*fabs(x) - alpha*alpha : slope = 2*alpha
}

double cost_function(const double* x)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr copy_visible_submap(new pcl::PointCloud<pcl::PointXYZI>() );
	pcl::copyPointCloud(*visible_submap, *copy_visible_submap);
    cv::Mat shift_depthmap(HEIGHT, WIDTH, CV_32F, cv::Scalar(SCENE_SIZE) );

    // camera to shift coordinates
    Eigen::Isometry3d T_cam_shift = Eigen::Isometry3d::Identity();
    Eigen::Map<const Eigen::Matrix<double, 6, 1> > T_se3(x); // shift amount
    Sophus::SE3 T_SE3 = Sophus::SE3::exp(T_se3); // T_SE3 is shift transformation
    T_cam_shift.rotate(T_SE3.rotation_matrix() );
    T_cam_shift.pretranslate(T_SE3.translation() ); 
	pcl::transformPointCloud (*copy_visible_submap, *copy_visible_submap, T_cam_shift.matrix() );

    std::vector<cv::Point3d> points_3d; // in shift coords
    std::vector<double> points3d_depth_mapping;
	for(int i = 0; i < copy_visible_submap->size(); i++) 
	{
		double x = copy_visible_submap->at(i).x;
		double y = copy_visible_submap->at(i).y;
		double z = copy_visible_submap->at(i).z;

		points_3d.push_back(cv::Point3d(x, y, z) );
        points3d_depth_mapping.push_back(z);
	}

    // project points: NO NEED TO TAKE DISTORTION INTO ACCOUNT SINCE DEPTH IMAGES ARE RECTIFIED
    std::vector<cv::Point2d> projectedPoints; 
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
    
    double residual = 0.0; 
    for(int i = 0; i < projectedPoints.size(); i++) // over all submap points
	{
		cv::Point2d projectedPt = projectedPoints[i];
        
        if( inBorder(projectedPt) )
        {
            int x = round(projectedPt.x);
		    int y = round(projectedPt.y);

            double error = points3d_depth_mapping[i] - depth_map.at<cv::Vec3f>(cv::Point(x, y) ).val[2];

            // double loss = huber_norm(error, 3) / g_variance[i];
            double loss = huber_norm(error, 3);
            if(loss > 15.0) 
                loss = 15.0; // error capped

            residual += loss;
        }	
        else
            residual += 7.0; // out of border penalty term: set to alpha^2
	}
    
    return residual;
}

bool cmp(const NMpoint& a, const NMpoint& b){
    return a.score < b.score;
}

void setVariance(std::vector<double>& variance)
{
    g_variance = variance;
}

class Nelder_Mead{
public:
    Nelder_Mead(double step=0.4, int max_iter=NELDER_MEAD_ITERS, double alpha=1.0, double gamma=2, double rho=-0.5, double sigma=0.5, double tolerence=0.001): 
        step(step), max_iter(max_iter), alpha(alpha), gamma(gamma), rho(rho), sigma(sigma), tolerence(tolerence)
    { multiple = 4.0; }

    void solve(double* para, int dim){
        // Make polytope
        std::vector<NMpoint> polytope;
        Eigen::VectorXd init_vec = Eigen::Map<Eigen::VectorXd>(para, dim);
        make_polytope(polytope, init_vec, dim);
        
        int iter = 0;
        while(1){
            if(iter >= max_iter) break;
            iter++;
            sort(polytope.begin(), polytope.end(), cmp);
            // Centroid
            Eigen::VectorXd x0(dim);
            x0.setZero();
            for(int i = 0; i < polytope.size() - 1; i++){
                x0 += polytope[i].vec;
            }
            x0 /= (polytope.size() - 1);
            double current_best_score = polytope[0].score;
            std::cout << "iter" << iter << "  current_best_score = " << current_best_score << "\n";
            double worst_score = polytope.back().score;
            double second_worst_score = polytope[polytope.size() - 2].score;

            // Reflection
            Eigen::VectorXd xr = x0 + alpha * (x0 - polytope.back().vec);
            double rscore = cost_function(xr.data());
            if(current_best_score <= rscore && rscore < second_worst_score){
                polytope.pop_back();
                polytope.push_back(NMpoint{xr, rscore});
                continue;
            }

            // Expansion
            if(rscore < current_best_score){
                Eigen::VectorXd xe = x0 + gamma * (x0 - polytope.back().vec);
                double escore = cost_function(xe.data());
                if(escore < rscore){
                    polytope.pop_back();
                    polytope.push_back(NMpoint{xe, escore});
                }
                else{
                    polytope.pop_back();
                    polytope.push_back(NMpoint{xr, rscore});
                }
                continue;
            }

            // Contraction
            Eigen::VectorXd xc = x0 + rho * (x0 - polytope.back().vec);
            double cscore = cost_function(xc.data());
            if(cscore < worst_score){
                polytope.pop_back();
                polytope.push_back(NMpoint{xc, cscore});
                continue;
            }

            // Shrink
            Eigen::VectorXd x1 = polytope[0].vec;
            for(int i = 1; i < polytope.size(); i++){
                polytope[i].vec = x1 + sigma * (polytope[i].vec - x1);
                polytope[i].score = cost_function(polytope[i].vec.data());
            }
            // Restart
            double rate = 2.0 * abs(polytope[0].score - polytope.back().score) / 
                        (abs(polytope[0].score) + abs(polytope.back().score));
            if(rate < tolerence){
                make_polytope(polytope, polytope[0].vec, dim);
                std::cout << "Restart\n";
            }
        }

        for(int i = 0; i < dim; i++){
            para[i] = polytope[0].vec(i);
        }
    }
private:
    void make_polytope(std::vector<NMpoint>& polytope, Eigen::VectorXd init_vec, int dim){
        polytope.clear();
        double score = cost_function(init_vec.data());
        polytope.push_back(NMpoint{init_vec, score});

        for(int i = 0; i < dim; i++){
            Eigen::VectorXd vec = init_vec;
            if(i < 3) vec(i) += step; // translation
            else if (i == 4) vec(i) += step / multiple; // rotation
            score = cost_function(vec.data());
            polytope.push_back(NMpoint{vec, score});
        }        
    }
private:
    const double step;
    const double alpha, gamma, rho, sigma;
    const double tolerence;
    double multiple;
    const int max_iter;
};
