#pragma once
#include <opencv2/highgui/highgui.hpp>

cv::Mat EstimateH(const std::vector<std::pair<cv::Point2f, cv::Point2f> > pointPairs);

void FitH(std::vector<std::pair<cv::Point2f, cv::Point2f> > pointPairs,
	std::vector<int>& inliersIdx,
	const cv::Mat& model,
	const double& threshold);

cv::Mat DenormalizeH(const cv::Mat H_norm, const cv::Mat& T1, const cv::Mat& T2);