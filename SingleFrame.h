#pragma  once

#include <iostream>
#include <opencv2/core/types.hpp>

struct SingleFrame {
    cv::Mat Frame;
    std::string FrameName;
};