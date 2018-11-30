#ifndef _knn_H_
#define _knn_H_
#include "opencv2\opencv.hpp"
#include <iostream>


cv::Ptr<cv::ml::KNearest> Train_KNN();
float  Predict_KNN(cv::Ptr<cv::ml::KNearest> model, cv::Mat sample);

#endif
