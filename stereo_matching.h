#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

/* abstract base class (interface) for correspondence search functors
*/
class CorrespSearchMethod {
public:
    // pure virtual function to get disparity map
    virtual cv::Mat operator() (cv::Mat img1, cv::Mat img2) = 0;
};

/* naive search derived class
*/
class NaiveSearch : public CorrespSearchMethod {
private:
	double SSD(cv::Mat subimg1, cv::Mat subimg2) {
		return cv::norm(subimg1, subimg2, cv::NORM_L2, cv::noArray());
	}

	double SAD(cv::Mat subimg1, cv::Mat subimg2) {
		return cv::norm(subimg1, subimg2, cv::NORM_L1, cv::noArray());
	}

	double NCC(cv::Mat subimg1, cv::Mat subimg2) {
		return subimg1.dot(subimg2) / (cv::norm(subimg1, cv::NORM_L2, cv::noArray()) * cv::norm(subimg2, cv::NORM_L2, cv::noArray()));
	}
	int dmax, win_size;
public:
	NaiveSearch(int dmax, int win_size) : dmax{ dmax }, win_size{ win_size }{}
	// implement
    cv::Mat operator() (cv::Mat img1, cv::Mat img2) override;
};

/* Patch Match derived class
*/
class PatchMatch : public CorrespSearchMethod {
private:
	cv::Mat random_init(int height, int width);
	void spatial_propagation_tl(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, int x, int y);
	void spatial_propagation_br(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, int x, int y);
	void view_propagation(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur, int x, int y);
	void plane_refinement(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, int x, int y);
	void top_left_iter(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur);
	void bottom_right_iter(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur);
	void compute_grad(cv::Mat& img1, cv::Mat& img2);
	void compute_weight(cv::Mat& img1, cv::Mat& img2);
	void post_processing(cv::Mat& disparity1, cv::Mat& disparity2);
	void wmf(cv::Mat& disparity1, cv::Mat& disparity2);
	cv::Mat to_disparity(cv::Mat& sur);
	double m(cv::Mat& main_img, cv::Mat& corr_img, int x, int y, cv::Vec3d abc);
	int dmax, win_size, nbh_size;
	bool is_main_left{true};
	cv::Mat2d grad_left;
	cv::Mat2d grad_right;
	cv::Mat weight_left;
	cv::Mat weight_right;
	cv::Mat1b valid_1;
	cv::Mat1b valid_2;
public:
	//dmax: maximum disparity, win_size: window size, nbh_size: neighborhood size in spatial propagation
	PatchMatch(int dmax, int win_size, int nbh_size) : dmax{ dmax }, win_size{ win_size }, nbh_size{ nbh_size }{}
	// implement
    cv::Mat operator() (cv::Mat img1, cv::Mat img2) override;
};

/* class StereoMatcher {
}; */
