#pragma  once

#include <iostream>
#include <opencv2/core/mat.hpp>

struct SingleFrame {
    cv::Mat Frame;
    std::string FrameName;
};