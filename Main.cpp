#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "MatrixReaderWriter.h"
#include "RANSAC.h"
#include "Homography.h"




//Tranformation of images
void transformImage(cv::Mat origImg, cv::Mat& newImage, cv::Mat tr, bool isPerspective) {
    cv::Mat invTr = tr.inv();
    const int WIDTH = origImg.cols;
    const int HEIGHT = origImg.rows;

    const int newWIDTH = newImage.cols;
    const int newHEIGHT = newImage.rows;



    for (int x = 0; x < newWIDTH; x++) {
        for (int y = 0; y < newHEIGHT; y++) {
            cv::Mat pt(3, 1, CV_32F);
            pt.at<float>(0, 0) = x;
            pt.at<float>(1, 0) = y;
            pt.at<float>(2, 0) = 1.0;

            cv::Mat ptTransformed = invTr * pt;
            if (isPerspective) ptTransformed = (1.0 / ptTransformed.at<float>(2, 0)) * ptTransformed;

            int newX = round(ptTransformed.at<float>(0, 0));
            int newY = round(ptTransformed.at<float>(1, 0));

            if ((newX >= 0) && (newX < WIDTH) && (newY >= 0) && (newY < HEIGHT)) newImage.at<cv::Vec3b>(y, x) = origImg.at<cv::Vec3b>(newY, newX);
        }
        std::cout << (double) x / newWIDTH * 100 << "%\r";
        std::cout.flush();
    }
}

int main(int argc, char** argv)
{
    //read all images in folder
    std::vector<cv::String> imageNames;
    std::vector<cv::String> featureNames;

    std::string folder = "images/2a/";
    std::string imageExtension = folder + "*.jpg";
    std::string featureExtension = folder + "*.txt";

    glob(imageExtension, imageNames, false);
    glob(featureExtension, featureNames, false);

    //read images:
    std::vector<cv::Mat> images;
    for (size_t i = 0; i < imageNames.size(); i++) {
        images.push_back(cv::imread(imageNames[i]));
    }

    //std::cout << featureNames.size() << std::endl;
    //std::cout << featureNames[0] << std::endl;
    //MatrixReaderWriter mrw(featureNames[0].c_str());
    //std::cout << mrw.data[0] << std::endl;
    //read features:
    for (int j = 0; j < featureNames.size(); j++) {

        std::pair<cv::Point2f, cv::Point2f> currPts;
        std::vector<cv::Point2f> p1, p2;
        MatrixReaderWriter mrw(featureNames[j].c_str());
        p1.reserve(mrw.rowNum);
        p2.reserve(mrw.rowNum);
        for (int i = 0; i < mrw.rowNum; i++) {
            currPts.first = cv::Point2f((float)mrw.data[i * mrw.columnNum], (float)mrw.data[i*mrw.columnNum + 1]);
            currPts.second = cv::Point2f((float)mrw.data[i * mrw.columnNum + 2], (float)mrw.data[i * mrw.columnNum + 3]);
            p1.push_back(currPts.first);
            p2.push_back(currPts.second);
        }

        cv::Mat T1 = cv::Mat(3, 3, CV_32F);
        cv::Mat T2 = T1;
        std::vector<cv::Point2f> p1_norm, p2_norm;

        /*NormalizeData(p1, p1_norm, T1);
        NormalizeData(p2, p2_norm, T2);*/

        std::vector<std::pair<cv::Point2f, cv::Point2f>> pointPairsVector;
        pointPairsVector.reserve(mrw.rowNum);
        for (int i = 0; i < mrw.rowNum; i++) {
            //currPts.first = p1_norm[i];
            //currPts.second = p2_norm[i];
            currPts.first = p1[i];
            currPts.second = p2[i];
            pointPairsVector.push_back(currPts);
        }

        std::vector<int> bestInliersIdx;
        cv::Mat H;
        RANSAC_LSQ(pointPairsVector, bestInliersIdx, H, 4, 3.0f, 0.99f, 1000);

        cv::Mat tempImage = cv::Mat::zeros(1.5 * images[0].size().height, 2 * images[0].size().width, images[0].type());

        transformImage(images[1], tempImage, cv::Mat::eye(3, 3, CV_32F), true);

        transformImage(images[0], tempImage, H, true);

        std::cout << H << std::endl;

        cv::imshow("Panorama", tempImage);
        cv::waitKey(0);

    }

    return 0;
}