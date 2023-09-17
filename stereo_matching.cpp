#include "stereo_matching.h"
#include <opencv2/core/mat.hpp>

cv::Mat NaiveSearch::operator() (cv::Mat img1, cv::Mat img2) {
    
    int image_height = img1.rows; //define internal variables: image width and image height
    int image_width = img1.cols;
    int num_total_pixel = image_height * image_width;

    cv::Mat Disparity = cv::Mat(image_height, image_width, CV_64F); //declare return variable
    auto roi_full_img = cv::Rect({}, img1.size()); //define the region of interest outbound of target image (for OpenCV intersection functionality use)

    int prnt_cnt{10}; //print count for progress check
    int prnt_chk_pt{std::static_cast<int>(num_total_pixel/prnt_cnt) + 1};

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
    return 0;
}