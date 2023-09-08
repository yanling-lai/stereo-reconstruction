#include <iostream>
#include <array>
#include "StereoImagePairContainer.h"
#include "stereo_matching.h"
#include <chrono>

int main(int argc, char** argv) {
    // argv format: [<Program Name>, DataDir, PairNames[0], PairNames[1], MatchingMethod]
    // argc: Have to == 5

    if (argc != 5) {
        std::cout << "Wrong arguments!\nArgument format: <Data Directory Path> <Left Frame File Name> <Right Frame File Name> <Matching Method>" << std::endl;
        return -1;
    }

    // Make sure this path points to the data folder
    std::string DataDir = argv[1];
    std::vector<std::string> PairNames;
    PairNames.push_back(argv[2]);
    PairNames.push_back(argv[3]);
    std::string MatchingMethod = argv[4];

    std::cout << "Reading in image pair..." << std::endl;
    StereoImagePairContainer *PairContainer;
    if (!PairContainer->Init(DataDir, PairNames)) {
        std::cout << "Failed to read in image pair!\nCheck data path and file names!" << std::endl;
        return -1;
    }

    // TODO: StereoMatcher s = StereoMatcher(PairContainer); => initialize stereomatcher with the stereo image pair
    // s.GetDisparity("PatchMatchStereo");
    // StereoReconstructor s = StereoReconstructor(*middleburyFrame_l, *middleburyFrame_r);

    int max_disparity = 100;
    int window_size = 96;

    if (MatchingMethod == "Naive") {
        auto run_start = std::chrono::high_resolution_clock::now();
        if (PairContainer->GetDisparity(new NaiveSearch(max_disparity, window_size))) {
            auto run_stop = std::chrono::high_resolution_clock::now();
            std::cout << "The Naive Search Function took " << std::chrono::duration_cast<std::chrono::seconds>(run_stop - run_start).count() << "second to completion." << std::endl;
            cv::imwrite(DataDir + "disparity.png", PairContainer->Disparity);
            return 0;
        } else {
            std::cout << "Failed." << std::endl;
            return -1;
        }
    } else if (MatchingMethod == "PatchMatchStereo") {
        auto run_start = std::chrono::high_resolution_clock::now();
        if (PairContainer->GetDisparity(new PatchMatch(max_disparity, window_size))) {
            auto run_stop = std::chrono::high_resolution_clock::now();
            std::cout << "The Naive Search Function took " << std::chrono::duration_cast<std::chrono::seconds>(run_stop - run_start).count() << "second to completion." << std::endl;
            cv::imwrite(DataDir + "disparity.png", PairContainer->Disparity);
            return 0;
        } else {
            std::cout << "Failed." << std::endl;
            return -1;
        }
    } else {
        std::cout << "Unsupported matching method! Check arguments." << std::endl;
        return -1;
    }

}
