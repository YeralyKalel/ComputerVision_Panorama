#pragma once
#include <opencv2/opencv.hpp>  



void NormalizeData(const std::vector<cv::Point2f>& inPts, std::vector<cv::Point2f>& outPts, cv::Mat& T);
