#pragma once

#include <opencv2/core.hpp>

size_t GetIterationNumber(
    const double& inlierRatio_,
    const double& confidence_,
    const size_t& sampleSize_);

std::vector<int> RandomPerm(
    const int& sampleSize,
    const int& dataSize);

void FitModel(
    const std::vector<std::pair<cv::Point2f, cv::Point2f> > pointPairs,
    std::vector<int>& inliersIdx,
    const cv::Mat& model,
    const double& threshold);

void FitModelLSQ(
    const std::vector<cv::Point3d>& points,
    const std::vector<int>& inliers,
    cv::Mat& planeModel);

void RANSAC_LSQ(
    const std::vector<std::pair<cv::Point2f, cv::Point2f>>& dataset,
    std::vector<int>& bestInliersIdx,
    cv::Mat& bestModel,
    const int sampleSize,
    double threshold,
    double confidence,
    int maxIter);