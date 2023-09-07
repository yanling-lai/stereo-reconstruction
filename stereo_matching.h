#pragma once
#include <opencv2/core/mat.hpp>
#include<opencv2/opencv.hpp>

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
	int dmax, win_size, nbh_size;
public:
	//dmax: maximum disparity, win_size: window size, nbh_size: neighborhood size in spatial propagation
	PatchMatch(int dmax, int win_size, int nbh_size) : dmax{ dmax }, win_size{ win_size }, nbh_size{ nbh_size }{}
	// implement
    cv::Mat operator() (cv::Mat img1, cv::Mat img2) override;
};

/* class StereoMatcher {
}; */