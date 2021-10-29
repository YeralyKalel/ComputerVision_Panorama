#include "Homography.h"
#include "Normalization.h"
#include <iostream>;
#include <math.h>  

cv::Mat EstimateH(const std::vector<std::pair<cv::Point2f, cv::Point2f> > pointPairs) {
    const int ptsNum = pointPairs.size();
    cv::Mat A(2 * ptsNum, 9, CV_32F);

    for (int i = 0; i < ptsNum; i++) {
        float u1 = pointPairs[i].first.x;
        float v1 = pointPairs[i].first.y;

        float u2 = pointPairs[i].second.x;
        float v2 = pointPairs[i].second.y;

        A.at<float>(2 * i, 0) = u1;
        A.at<float>(2 * i, 1) = v1;
        A.at<float>(2 * i, 2) = 1.0f;
        A.at<float>(2 * i, 3) = 0.0f;
        A.at<float>(2 * i, 4) = 0.0f;
        A.at<float>(2 * i, 5) = 0.0f;
        A.at<float>(2 * i, 6) = -u2 * u1;
        A.at<float>(2 * i, 7) = -u2 * v1;
        A.at<float>(2 * i, 8) = -u2;

        A.at<float>(2 * i + 1, 0) = 0.0f;
        A.at<float>(2 * i + 1, 1) = 0.0f;
        A.at<float>(2 * i + 1, 2) = 0.0f;
        A.at<float>(2 * i + 1, 3) = u1;
        A.at<float>(2 * i + 1, 4) = v1;
        A.at<float>(2 * i + 1, 5) = 1.0f;
        A.at<float>(2 * i + 1, 6) = -v2 * u1;
        A.at<float>(2 * i + 1, 7) = -v2 * v1;
        A.at<float>(2 * i + 1, 8) = -v2;

    }

    cv::Mat eVecs(9, 9, CV_32F), eVals(9, 9, CV_32F);
    //std::cout << A << std::endl;
    eigen(A.t() * A, eVals, eVecs);

    cv::Mat H(3, 3, CV_32F);
    for (int i = 0; i < 9; i++) H.at<float>(i / 3, i % 3) = eVecs.at<float>(8, i);

    //Normalize:
    H = H * (1.0 / H.at<float>(2, 2));
    return H;
}

void FitH(std::vector<std::pair<cv::Point2f, cv::Point2f>> pointPairs,
    std::vector<int>& inliersIdx,
    const cv::Mat& H,
    const double& threshold) {

    double _threshold = pow(threshold, 2);
    inliersIdx.clear();
    cv::Point2f pe2; //estimated points using H
    double hom;
    double h11, h12, h13, h21, h22, h23, h31, h32, h33, p1x, p1y, p2x, p2y;

    h11 = H.at<float>(0, 0);
    h12 = H.at<float>(0, 1);
    h13 = H.at<float>(0, 2);
    h21 = H.at<float>(1, 0);
    h22 = H.at<float>(1, 1);
    h23 = H.at<float>(1, 2);
    h31 = H.at<float>(2, 0);
    h32 = H.at<float>(2, 1);
    h33 = H.at<float>(2, 2);

    //std::cout << H << std::endl;

    for (int i = 0; i < pointPairs.size(); i++) {
        p1x = pointPairs[i].first.x;
        p1y = pointPairs[i].first.y;
        p2x = pointPairs[i].second.x;
        p2y = pointPairs[i].second.y;

        hom = h31 * p1x + h32 * p1y + h33;

        pe2.x = (h11 * p1x + h12 * p1y + h13) / hom;
        pe2.y = (h21 * p1x + h22 * p1y + h23) / hom;

        double error = pow(pe2.x - p2x, 2) + pow(pe2.y - p2y, 2);
        if (error < _threshold) {
            inliersIdx.push_back(i);
        }
    }
}

cv::Mat DenormalizeH(const cv::Mat H_norm, const cv::Mat& T1, const cv::Mat& T2) {
    return T2.inv() * H_norm * T1;
}