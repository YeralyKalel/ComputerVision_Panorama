#include "RANSAC.h"
#include "Homography.h"
#include "Normalization.h"
#include <iostream>

//Functions:
size_t GetIterationNumber(
    const double& inlierRatio_,
    const double& confidence_,
    const size_t& sampleSize_)
{
    std::cout << "Inlier ratio is " << inlierRatio_ << std::endl;
    double a =
        log(1.0 - confidence_);
    double b =
        log(1.0 - std::pow(inlierRatio_, sampleSize_));

    if (abs(b) < std::numeric_limits<double>::epsilon())
        return std::numeric_limits<size_t>::max();

    return a / b;
}

std::vector<int> RandomPerm(
    const int& sampleSize,
    const int& dataSize) {
    std::vector<int> result;
    for (size_t sampleIdx = 0; sampleIdx < sampleSize && sampleIdx < dataSize; sampleIdx++) {
        int val;
        bool isFound;
        do {
            val = rand() % (dataSize - 1);
            isFound = false;
            for (size_t i = 0; i < result.size(); i++) {
                if (val == result[i]) {
                    isFound = true;
                    break;
                }
            }
        } while (isFound);
        result.push_back(val);
    }
    return result;
}

void FitModel(
    const std::vector<std::pair<cv::Point2f, cv::Point2f> > pointPairs,
    std::vector<int>& inliersIdx,
    const cv::Mat& model,
    const double& threshold) {
    FitH(pointPairs, inliersIdx, model, threshold);
}

//void FitModelLSQ(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& dataset,
//    const std::vector<int>& inliersIdx,
//    cv::Mat& model) {
//    std::vector<cv::Point3d> normalizedPoints;
//    cv::Point3d masspoint(0, 0, 0);
//
//    std::vector<cv::Point2f> p1, p2, p1_norm, p2_norm;
//
//    for (int i = 0; i < inliersIdx.size(); i++) {
//        p1.push_back(dataset[inliersIdx[i]].first);
//        p2.push_back(dataset[inliersIdx[i]].second);
//    }
//
//    cv::Mat T1, T2;
//    NormalizeData(p1, p1_norm, T1);
//    NormalizeData(p2, p2_norm, T2);
//
//    std::vector<std::pair<cv::Point2f, cv::Point2f>> inlierPairs;
//    std::pair<cv::Point2f, cv::Point2f> currPts;
//    for (int i = 0; i < inliersIdx.size(); i++) {
//        currPts.first = p1_norm[i];
//        currPts.second = p2_norm[i];
//        inlierPairs.push_back(currPts);
//    }
//
//    cv::Mat H_norm = EstimateH(inlierPairs);
//    model = DenormalizeH(H_norm, T1, T2);
//}

void RANSAC_LSQ(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& dataset,
    std::vector<int>& bestInliersIdx,
    cv::Mat& bestModel,
    const int kSampleSize,
    double threshold,
    double confidence,
    int maxIter) {

    if (dataset.size() < kSampleSize) {
        std::cout << "Not enough data for RANSAC." << std::endl;
        return;
    }

    std::cout << "RANSAC is running." << std::endl;

    std::vector<std::pair<cv::Point2f, cv::Point2f>> dataset_norm;
    std::vector<cv::Point2f> p1, p2, p1_norm, p2_norm;
    cv::Mat T1, T2;
    for (int i = 0; i < dataset.size(); i++) {
        p1.push_back(dataset[i].first);
        p2.push_back(dataset[i].second);
    }
    NormalizeData(p1, p1_norm, T1);
    NormalizeData(p2, p2_norm, T2);
    std::pair<cv::Point2f, cv::Point2f> currPts;
    for (int i = 0; i < dataset.size(); i++) {
        currPts.first = p1_norm[i];
        currPts.second = p2_norm[i];
        dataset_norm.push_back(currPts);
    }

    size_t maxIter_ = maxIter;

    size_t iter = 0;
    size_t bestInlierNumber = 0;

    std::vector<int> sampleIdx(kSampleSize);
    std::vector<std::pair<cv::Point2f, cv::Point2f>> sample(kSampleSize);

    std::vector<int> inliersIdx;

    while (iter++ < maxIter_) {

        //Select random points
        sample.clear();
        sampleIdx = RandomPerm(kSampleSize, dataset.size());
        for (size_t i = 0; i < kSampleSize; i++) {
            sample.push_back(dataset_norm[sampleIdx[i]]);
        }

        //Create model:
        cv::Mat H = EstimateH(sample);
        H = DenormalizeH(H, T1, T2);
        //Fit model:
        FitModel(dataset, inliersIdx, H, threshold);

        //Check the model:
        if (inliersIdx.size() > bestInlierNumber) {
            bestModel = H;
            bestInliersIdx = inliersIdx;
            bestInlierNumber = inliersIdx.size();

            // Update the maximum iteration number
            maxIter_ = GetIterationNumber(
                static_cast<double>(bestInlierNumber) / static_cast<double>(dataset.size()),
                confidence,
                kSampleSize);

            printf("Inlier number = %d\tMax iterations = %d\n", (int)bestInliersIdx.size(), (int)maxIter_);
        }

        std::cout << (double)iter / maxIter_ * 100 << "%\r";
        std::cout.flush();
    }
    std::cout << "RANSAC is finished." << std::endl;
}
