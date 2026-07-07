#include "stereo_matching.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>
#include <utility>
#include <string>

double clr_thresh{10.0};
double grd_thresh{2.0};
double Gamma{10.0};
double Alpha{0.9};

cv::Mat NaiveSearch::operator() (cv::Mat img1, cv::Mat img2) {
    
    int image_height = img1.rows; //define internal variables: image width and image height
    int image_width = img1.cols;
    int num_total_pixel = image_height * image_width;

    cv::Mat Disparity = cv::Mat(image_height, image_width, CV_64F); //declare return variable
    auto roi_full_img = cv::Rect({}, img1.size()); //define the region of interest outbound of target image (for OpenCV intersection functionality use)

    int prnt_cnt{10}; //print count for progress check
    int prnt_chk_pt = static_cast<int>(num_total_pixel/prnt_cnt + 1);

    for (int i{0}; i < num_total_pixel; ++i) {
        int row{i / image_width}; //define current row
        int col{i % image_width}; //define current column
        double min{100000.0};
        int min_pos{col};
        auto roi_sliding_window1 = cv::Rect(col - win_size / 2, row - win_size / 2, win_size, win_size); //define sliding window for target image
        auto roi_img1 = roi_full_img & roi_sliding_window1; //define intersection of target image (image1) and target sliding window

        for (int j{std::max(col - dmax, 0)}; j <= col; ++j) {
            double c{0.0}; //variable for cost value
            auto roi_sliding_window2 = cv::Rect(j - win_size / 2, row - win_size / 2, win_size, win_size); //define sliding window for source image
            auto roi_img2 = roi_full_img & roi_sliding_window2; //define intersection of source image (image2) and source sliding window
            
            //compute final cropping bound for target and source images
            auto temp = roi_img2 - roi_sliding_window2.tl() + roi_sliding_window1.tl();
            roi_img1 = temp & roi_img1;
            roi_img2 = roi_img1 - roi_sliding_window1.tl() + roi_sliding_window2.tl();

            //crop out patches from target and source images and evaluate their differences
            cv::Mat patch_target = img1(roi_img1);
            cv::Mat patch_source = img2(roi_img2);

            //compute cost and decide if it's new min
            c = SSD(patch_target, patch_source);
            min_pos = (c < min) ? j : min_pos;
            min = (c < min) ? c : min;
        }
        //compute disparity
        Disparity.at<double>(row, col) = col - min_pos;

        //print out progress when every approx 10% of pixels are processed
        if ((i == prnt_chk_pt) || (i == num_total_pixel-1)) {
            std::cout << "Naive Search: done for " << (10-prnt_cnt+1)*10 << "%" << " of " << "100%" << " image." << std::endl;
            prnt_chk_pt += prnt_chk_pt;
            prnt_cnt--;
        }
    }
    return Disparity;
}

cv::Mat PatchMatch::operator() (cv::Mat img1, cv::Mat img2) {
	int height = img1.rows;
	int width = img1.cols;
	cv::Mat surface1 = random_init(height, width);
	cv::Mat surface2 = random_init(height, width);
	compute_weight(img1, img2);
	compute_grad(img1, img2);

	// run 3 iteration
	for (int i{ 0 }; i < 3; ++i) {
		// odd iteration: from bottom right to top left
		if (i % 2) {
			// left image
			is_main_left = true;
			bottom_right_iter(img1, img2, surface1, surface2);

			// right image
			is_main_left = false;
			bottom_right_iter(img2, img1, surface2, surface1);
		} else {
			// even iteration: start from top left to bottom right
			// left image
			is_main_left = true;
			top_left_iter(img1, img2, surface1, surface2);

			// right image
			is_main_left = false;
			top_left_iter(img2, img1, surface2, surface1);
		}
	}

	// convert plane parameters to disparity value
	cv::Mat disparity1 = to_disparity(surface1);
	cv::Mat disparity2 = to_disparity(surface2);

	// do post processing
	post_processing(disparity1, disparity2);

	// apply weighted median filter
	wmf(disparity1, disparity2);
	return disparity1;
}


cv::Mat PatchMatch::random_init(int height, int width) {
	// random initialize both surfaces (plane parameters)
	std::cout << "random initializing..." << std::endl;
	cv::Mat surface(height, width, CV_64FC3);
	cv::RNG rd_gen;
	const int RAND_HALF = RAND_MAX/2;

	for(int i{0}; i < height; ++i) {
		for(int j{0}; j < width; ++j) {
			double z = rd_gen.uniform(0.0, (double)dmax);
			double nx = ((double)std::rand() - RAND_HALF) / RAND_HALF;
			double ny = ((double)std::rand() - RAND_HALF) / RAND_HALF;
			double nz = ((double)std::rand() - RAND_HALF) / RAND_HALF;

			cv::Vec3d normal(nx, ny, nz);
			cv::normalize(normal, normal);
			cv::Vec3d abc;
			abc[0] = -normal[0]/normal[2];
			abc[1] = -normal[1]/normal[2];
			abc[2] = (j * normal[0] + i * normal[1] + z * normal[2]) / normal[2];
			surface.at<cv::Vec3d>(i, j) = abc;
		}
	}
	return surface;
}

void PatchMatch::spatial_propagation_tl(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, int x, int y) {
	//std::cout << "spatial propagating..." << std::endl;
	int wid_bd{0};
	int hgh_bd{0};
	double min_m{10000.0};
	double cst{0.0};
	int min_x{x};
	int min_y{y};
	cv::Vec3d abc;
	cv::Vec3d tmp;

	// bound check, set maximum shift of pixel x, y to proper bound to avoid segmentation fault
	wid_bd = ((x+nbh_size) < main_img.cols) ? nbh_size : main_img.cols-x;
	hgh_bd = ((y+nbh_size) < main_img.rows) ? nbh_size : main_img.rows-y;

	for(int i{0}; i < hgh_bd; ++i) {
		for(int j{0}; j < wid_bd; ++j) {
			abc = main_sur.at<cv::Vec3d>(y+i, x+j);

			cst = m(main_img, corr_img, x, y, abc);
			if(cst < min_m) {
				min_m = cst;
				min_x = x+j;
				min_y = y+i;
			}
		}
	}
	main_sur.at<cv::Vec3d>(y, x) = main_sur.at<cv::Vec3d>(min_y, min_x);
}

void PatchMatch::spatial_propagation_br(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, int x, int y) {
	//std::cout << "spatial propagating..." << std::endl;
	int wid_bd{0};
	int hgh_bd{0};
	double min_m{10000.0};
	double cst{0.0};
	int min_x{x};
	int min_y{y};
	cv::Vec3d abc;
	cv::Vec3d tmp;

	// bound check, set maximum shift of pixel x, y to proper bound to avoid segmentation fault
	wid_bd = ((x-nbh_size+1) >= 0) ? nbh_size : x+1;
	hgh_bd = ((y-nbh_size+1) >= 0) ? nbh_size : y+1;

	for(int i{0}; i < hgh_bd; ++i) {
		for(int j{0}; j < wid_bd; ++j) {
			abc = main_sur.at<cv::Vec3d>(y-i, x-j);

			cst = m(main_img, corr_img, x, y, abc);
			if(cst < min_m) {
				min_m = cst;
				min_x = x-j;
				min_y = y-i;
			}
		}
	}
	main_sur.at<cv::Vec3d>(y, x) = main_sur.at<cv::Vec3d>(min_y, min_x);
}

void PatchMatch::view_propagation(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur, int x, int y) {
	//std::cout << "view propagating..." << std::endl;
	// if(is_main_left) {
	int bnd = (is_main_left) ? x : main_img.cols-1-x;
	// bnd = ((x - dmax) < 0) ? x : dmax; //make sure no segmentation fault when near image left border
	int sign = (is_main_left) ? 1 : -1;
	cv::Vec3d abc;
	int corr_x{0};
	int disp{0};
	double cst{0.0};
	double min_m;
	int min_x{x};

	abc = main_sur.at<cv::Vec3d>(y, x);
	min_m = m(main_img, corr_img, x, y, abc);
	
	for(int i{0}; i <= bnd; ++i) {
		corr_x = x - sign * i;
		abc = corr_sur.at<cv::Vec3d>(y, corr_x);
		disp = abc[0] * corr_x + abc[1] * y + abc[2] + 0.5;

		if(disp == i) {
			cst = m(main_img, corr_img, x, y, abc);
			if(cst < min_m) {
				min_m = cst;
				min_x = corr_x;
			}
		}
	}
	
	main_sur.at<cv::Vec3d>(y, x) = corr_sur.at<cv::Vec3d>(y, min_x);
}

void PatchMatch::plane_refinement(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, int x, int y) {
	//std::cout << "plane refining..." << std::endl;
	double max_delta_z = dmax/2;
	double max_delta_n = 1;
	cv::Vec3d abc;
	abc = main_sur.at<cv::Vec3d>(y, x);
	double min_m{0.0};
	min_m = m(main_img, corr_img, x, y, abc);

	while(max_delta_z >= 0.1) {
		std::random_device rnd;
		std::mt19937 rnd_gen(rnd());

		std::uniform_real_distribution<> rnd_z(-max_delta_z, max_delta_z);
		std::uniform_real_distribution<> rnd_n(-max_delta_n, max_delta_n);

		// z + delta_z
		abc = main_sur.at<cv::Vec3d>(y, x);
		double z{0.0};
		z = abc[0] * x + abc[1] * y + abc[2];
		double delta_z{0.0};
		delta_z = rnd_z(rnd_gen);
		z += delta_z;

		// normal + delta_n
		cv::Vec3d normal;
		normal[0] = -abc[0];
		normal[1] = -abc[1];
		normal[2] = 1.0;
		cv::normalize(normal, normal);
		cv::Vec3d delta_n;
		delta_n[0] = rnd_n(rnd_gen);
		delta_n[1] = rnd_n(rnd_gen);
		delta_n[2] = rnd_n(rnd_gen);
		normal += delta_n;
		cv::normalize(normal, normal);

		// new abc
		abc[0] = -normal[0]/normal[2];
		abc[1] = -normal[1]/normal[2];
		abc[2] = (x * normal[0] + y * normal[1] + z * normal[2]) / normal[2];
		double new_m{0.0};
		new_m = m(main_img, corr_img, x, y, abc);

		// compare
		if(new_m < min_m) {
			min_m = new_m;
			main_sur.at<cv::Vec3d>(y, x) = abc;
		}

		max_delta_z /= 2.0;
		max_delta_n /= 2.0;
	}
}

void PatchMatch::compute_grad(cv::Mat& img1, cv::Mat& img2) {
	// store precomputed gradients for both images
	std::cout << "computing gray scale gradient..." << std::endl;
	grad_left = cv::Mat2d(img1.rows, img1.cols);
	grad_right = cv::Mat2d(img2.rows, img2.cols);

	cv::Mat left_gray, right_gray, left_grad_x, left_grad_y, right_grad_x, right_grad_y;
	cv::cvtColor(img1, left_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img2, right_gray, cv::COLOR_BGR2GRAY);
	cv::Sobel(left_gray, left_grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(left_gray, left_grad_y, CV_64F, 0, 1, 3);
	cv::Sobel(right_gray, right_grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(right_gray, right_grad_y, CV_64F, 0, 1, 3);
	left_grad_x = left_grad_x/8.0;
	left_grad_y = left_grad_y/8.0;
	right_grad_x = right_grad_x/8.0;
	right_grad_y = right_grad_y/8.0;

	for(int i{0}; i < img1.rows; ++i) {
		for(int j{0}; j < img1.cols; ++j) {
			grad_left.at<cv::Vec2d>(i, j)[0] = left_grad_x.at<double>(i, j);
			grad_left.at<cv::Vec2d>(i, j)[1] = left_grad_y.at<double>(i, j);
			grad_right.at<cv::Vec2d>(i, j)[0] = right_grad_x.at<double>(i, j);
			grad_right.at<cv::Vec2d>(i, j)[1] = right_grad_y.at<double>(i, j);
		}
	}
}

void PatchMatch::compute_weight(cv::Mat& img1, cv::Mat& img2) {
	// store precomputed weights for both images
	std::cout << "computing exponential weight..." << std::endl;
	int arr[4] = {img1.rows, img1.cols, win_size, win_size};
	weight_left = cv::Mat(4, arr, CV_64F);
	weight_right = cv::Mat(4, arr, CV_64F);
	cv::Mat img1_d;
	cv::Mat img2_d;
	img1.convertTo(img1_d, CV_64FC3);
	img2.convertTo(img2_d, CV_64FC3);

	for(int i{0}; i < img1_d.rows; ++i) {
		for(int j{0}; j < img1_d.cols; ++j) {
			for(int p_i{i-win_size/2}; p_i <= i+win_size/2; ++p_i) {
				for(int p_j{j-win_size/2}; p_j <=j+win_size/2; ++p_j) {
					//check inside
					if((0 <= p_i) && (p_i < img1_d.rows) && (0 <= p_j) && (p_j < img1_d.cols)) {
						weight_left.at<double>(cv::Vec<int, 4> {i, j, p_i-i+win_size/2, p_j-j+win_size/2}) = std::exp(-cv::norm(img1_d.at<cv::Vec3d>(i, j) - img1_d.at<cv::Vec3d>(p_i, p_j), cv::NORM_L1)/Gamma);
						weight_right.at<double>(cv::Vec<int, 4> {i, j, p_i-i+win_size/2, p_j-j+win_size/2}) = std::exp(-cv::norm(img2_d.at<cv::Vec3d>(i, j) - img2_d.at<cv::Vec3d>(p_i, p_j), cv::NORM_L1)/Gamma);
					}
				}
			}
		}
	}
}

double PatchMatch::m(cv::Mat& main_img, cv::Mat& corr_img, int x, int y, cv::Vec3d abc) {
	// cost function
	double m{0.0};
	double disp;
	double match_x{0.0};
	int match_x_int{0};
	double lin_po_w{0.0};
	double w{0.0};
	cv::Vec3d main_clr, corr_clr;
	cv::Vec2d main_g, corr_g;
	cv::Mat main_img_d;
	cv::Mat corr_img_d;

	main_img.convertTo(main_img_d, CV_64FC3);
	corr_img.convertTo(corr_img_d, CV_64FC3);

	for(int i{y-win_size/2}; i <= y+win_size/2; ++i) {
		for(int j{x-win_size/2}; j < x+win_size/2; ++j) {
			// check inside
			if((0 > i) || (i >= main_img_d.rows) || (0 > j) || (j >= main_img_d.cols)) {
				continue;
			}
			// compute disparity
			disp = abc[0] * j + abc[1] * i + abc[2];

			// penalty for infeasible disparity
			if((disp < 0) || (disp > dmax)) {
				m+=120;
			} else {
				match_x = (is_main_left) ? x-disp : x+disp;
				match_x_int = (int)match_x;
				lin_po_w = 1.0-(match_x-(double)match_x_int); // match_x_int's weight

				if(match_x_int < 0) {
					match_x_int = 0;
				} else if(match_x_int > (main_img_d.cols-2)) {
					match_x_int = main_img_d.cols-2;
				}

				// linear interpolation of matching points color values and gradients
				main_clr = main_img_d.at<cv::Vec3d>(y, x);
				corr_clr = lin_po_w * corr_img_d.at<cv::Vec3d>(y, match_x_int) + (1-lin_po_w) * corr_img_d.at<cv::Vec3d>(y, match_x_int+1);

				if(is_main_left) {
					main_g = grad_left.at<cv::Vec2d>(y, x);
					corr_g = lin_po_w * grad_right.at<double>(y, match_x_int) + (1-lin_po_w) * grad_right.at<double>(y, match_x_int+1);
					w = weight_left.at<double>(cv::Vec<int, 4> {y, x, i-y+win_size/2, j-x+win_size/2});
				} else {
					main_g = grad_right.at<cv::Vec2d>(y, x);
					corr_g = lin_po_w * grad_left.at<double>(y, match_x_int) + (1-lin_po_w) * grad_left.at<double>(y, match_x_int+1);
					w = weight_right.at<double>(cv::Vec<int, 4> {y, x, i-y+win_size/2, j-x+win_size/2});
				}

				m += w * ((1-Alpha) * std::min(cv::norm(main_clr-corr_clr, cv::NORM_L1), clr_thresh) + Alpha * std::min(cv::norm(main_g-corr_g, cv::NORM_L1), grd_thresh));

			}
		}
	}
	return m;
}

void PatchMatch::top_left_iter(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur) {
	int height = main_img.rows;
	int width = main_img.cols;

	if(is_main_left) {
		std::cout << "top left, left img" << std::endl;
	} else {
		std::cout << "top left, right img" << std::endl;	
	}

	for (int i{ 0 }; i < width * height; ++i) {
		int row{ i / width };
		int col{ i % width };

		spatial_propagation_tl(main_img, corr_img, main_sur, col, row);
		view_propagation(main_img, corr_img, main_sur, corr_sur, col, row);
		plane_refinement(main_img, corr_img, main_sur, col, row);
	}
}

void PatchMatch::bottom_right_iter(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur) {
	int height = main_img.rows;
	int width = main_img.cols;

	if(is_main_left) {
		std::cout << "bottom right, left img" << std::endl;
	} else {
		std::cout << "bottom right, right img" << std::endl;	
	}

	for (int i{ (width * height - 1) }; i >= 0; --i) {
		int row{ i / width };
		int col{ i % width };

		spatial_propagation_br(main_img, corr_img, main_sur, col, row);
		view_propagation(main_img, corr_img, main_sur, corr_sur, col, row);
		plane_refinement(main_img, corr_img, main_sur, col, row);
	}
}

cv::Mat PatchMatch::to_disparity(cv::Mat& sur) {
	cv::Mat disp = cv::Mat(sur.rows, sur.cols, CV_64F);
	cv::Vec3d abc;
	for(int i{0}; i < sur.rows; ++i) {
		for(int j{0}; j < sur.cols; ++j) {
			abc = sur.at<cv::Vec3d>(i, j);
			disp.at<double>(i, j) = j * abc[0] + i * abc[1] + abc[2];
		}
	}
	return disp;
}

void PatchMatch::post_processing(cv::Mat& disparity1, cv::Mat& disparity2) {
	std::cout << "start post processing..." << std::endl;
	// storage for validity state for every pixel
	valid_1 = cv::Mat1b(disparity1.rows, disparity1.cols, (unsigned char)false);
	valid_2 = cv::Mat1b(disparity2.rows, disparity2.cols, (unsigned char)false);

	// check validity for every disparity
	for(int i{0}; i < disparity1.rows; ++i) {
		for(int j{0}; j < disparity1.cols; ++j) {
			int corr_x1 = std::max(0.0, std::min((double)disparity1.cols, j - disparity1.at<double>(i, j)));
			int corr_x2 = std::max(0.0, std::min((double)disparity2.cols, j + disparity2.at<double>(i, j)));

			valid_1.at<unsigned char>(i, j) = (std::fabs(disparity1.at<double>(i, j) - disparity2.at<double>(i, corr_x1)) <= 1.0);
			valid_2.at<unsigned char>(i, j) = (std::fabs(disparity2.at<double>(i, j) - disparity1.at<double>(i, corr_x2)) <= 1.0);
		}
	}

	// replace invalid disparities by nearest left or right valid neighbors disparities
	for(int i{0}; i < disparity1.rows; ++i) {
		for(int j{0}; j < disparity1.cols; ++j) {
			if(!valid_1.at<unsigned char>(i, j)) {
				int loc_lft{j - 1};
				while((!valid_1.at<unsigned char>(i, loc_lft)) && (loc_lft >= 0)) {
					--loc_lft;
				}

				int loc_rgt{j + 1};
				while((!valid_1.at<unsigned char>(i, loc_rgt)) && (loc_rgt < disparity1.cols)) {
					++loc_rgt;
				}

				if((loc_lft >= 0) && (loc_rgt < disparity1.cols)) {
					disparity1.at<double>(i, j) = (disparity1.at<double>(i, loc_lft) <= disparity1.at<double>(i, loc_rgt)) ? disparity1.at<double>(i, loc_lft) : disparity1.at<double>(i, loc_rgt);
				} else if(loc_lft >= 0) {
					disparity1.at<double>(i, j) = disparity1.at<double>(i, loc_lft);
				} else if(loc_rgt < disparity1.cols) {
					disparity1.at<double>(i, j) = disparity1.at<double>(i, loc_rgt);
				}
			}

			if(!valid_2.at<unsigned char>(i, j)) {
				int loc_lft{j - 1};
				while((!valid_2.at<unsigned char>(i, loc_lft)) && (loc_lft >= 0)) {
					--loc_lft;
				}

				int loc_rgt{j + 1};
				while((!valid_2.at<unsigned char>(i, loc_rgt)) && (loc_rgt < disparity2.cols)) {
					++loc_rgt;
				}

				if((loc_lft >= 0) && (loc_rgt < disparity2.cols)) {
					disparity2.at<double>(i, j) = (disparity2.at<double>(i, loc_lft) <= disparity2.at<double>(i, loc_rgt)) ? disparity2.at<double>(i, loc_lft) : disparity2.at<double>(i, loc_rgt);
				} else if(loc_lft >= 0) {
					disparity2.at<double>(i, j) = disparity2.at<double>(i, loc_lft);
				} else if(loc_rgt < disparity2.cols) {
					disparity2.at<double>(i, j) = disparity2.at<double>(i, loc_rgt);
				}
			}
		}
	}
}

void PatchMatch::wmf(cv::Mat& disparity1, cv::Mat& disparity2) {
	// weighted median filter
	for(int i{0}; i < disparity1.rows; ++i) {
		for(int j{0}; j < disparity1.cols; ++j) {
			// check invalid center
			if(!valid_1.at<unsigned char>(i, j)) {
				double w_total{0.0};
				double w{0.0};
				std::vector<std::pair<double, double>> disp_w;
				disp_w.reserve(win_size);
				for(int k{i - win_size/2}; k <= i + win_size/2; ++k) {
					for(int l{j - win_size/2}; l <= j + win_size/2; ++l) {
						if(((0 <= k) && (k < disparity1.rows) && (0 <= l) && (l < disparity1.cols)) && (valid_1.at<unsigned char>(k, l))) {
							// collect valid neighbors disparity and weight
							disp_w.push_back(std::make_pair(disparity1.at<double>(k, l), weight_left.at<double>(cv::Vec<int, 4> {i, j, k-i+win_size/2, l-j+win_size/2})));
							w_total += weight_left.at<double>(cv::Vec<int, 4> {i, j, k-i+win_size/2, l-j+win_size/2});
						}
					}
				}

				std::sort(disp_w.begin(), disp_w.end());
				for(auto p=disp_w.begin(); p<disp_w.end(); ++p) {
					w += p->second;
					if(w >= w_total/2.0) {
						if(p==disp_w.begin()) {
							disparity1.at<double>(i, j) = p->first;
							break;
						} else {
							disparity1.at<double>(i, j) = ((p-1)->first + p->first)/2.0;
							break;
						}
					}
				}
			}

			// check invalid center
			if(!valid_2.at<unsigned char>(i, j)) {
				double w_total{0.0};
				double w{0.0};
				std::vector<std::pair<double, double>> disp_w;
				disp_w.reserve(win_size);
				for(int k{i - win_size/2}; k <= i + win_size/2; ++k) {
					for(int l{j - win_size/2}; l <= j + win_size/2; ++l) {
						if(((0 <= k) && (k < disparity2.rows) && (0 <= l) && (l < disparity2.cols)) && (valid_2.at<unsigned char>(k, l))) {
							// collect valid neighbors disparity and weight
							disp_w.push_back(std::make_pair(disparity2.at<double>(k, l), weight_right.at<double>(cv::Vec<int, 4> {i, j, k-i+win_size/2, l-j+win_size/2})));
							w_total += weight_right.at<double>(cv::Vec<int, 4> {i, j, k-i+win_size/2, l-j+win_size/2});
						}
					}
				}

				std::sort(disp_w.begin(), disp_w.end());
				for(auto p=disp_w.begin(); p<disp_w.end(); ++p) {
					w += p->second;
					if(w >= w_total/2.0) {
						if(p==disp_w.begin()) {
							disparity2.at<double>(i, j) = p->first;
							break;
						} else {
							disparity2.at<double>(i, j) = ((p-1)->first + p->first)/2.0;
							break;
						}
					}
				}
			}
		}
	}
}