#pragma once

#include <vector>
#include <map>
#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <opencv2/core/types.hpp>
#include "SingleFrame.h"
#include "stereo_matching.h"


class StereoImagePairContainer {
public:

    StereoImagePairContainer(){

    }

    ~StereoImagePairContainer() {
        for (int i = 0; i < Frames.size(); i++) {

            SingleFrame *e = Frames.at(i);

            if (e->frame.empty() == false) {
                e->frame.release();
            }
            delete e;
        }
    }

    bool Init(const std::string &input_DataDir, const std::vector<std::string> &input_PairNames) {

        DataDir = input_DataDir;
        PairNames = input_PairNames;

        try {

            for (int i = 0; i < 2; ++i) {
                SingleFrame *frame;
                frame->FrameName = PairNames.at(i);
                frame->Frame = cv::imread(DataDir + PairNames.at(i), 1);
                Frames.push_back(frame);
            }
        }
        catch (...){
            std::cerr<< "Image file not found." << std::endl;
            return false;
        }
        return true;
    }

    bool GetDisparity(CorrespSearchMethod *c) {
        
        try {
            Disparity = (*c)(Frames.at(0)->Frame, Frames.at(1)->Frame);
        }
        catch (...) {
            std::cerr<< "Failed to establish disparity map." << std::endl;
            return false;
        }
        return true;
    }

private:

    std::vector<SingleFrame *> Frames; // Vector of stereo frames.
    std::string DataDir; // Directory to stereo image data.
    std::vector<std::string> PairNames; // Vector of frame names.
    cv::Mat Disparity;

};
