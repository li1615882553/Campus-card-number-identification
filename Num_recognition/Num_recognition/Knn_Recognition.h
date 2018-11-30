#pragma once
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include <iostream>

#include "Processing_function.h"
#include "knn.h"

class Knn_Recognition
{
public:
	Knn_Recognition();
	Knn_Recognition(std::string Image_path);
	~Knn_Recognition();
	void Set_Knn_Recognition_Image_path(std::string Image_path_)
	{
		Image_path = Image_path_;
	}
	void Knn_Training();                   //训练得到Knn的指针
	void detect(cv::Mat img, cv::Ptr<cv::ml::KNearest> model);
	void Solve();


private:
	std::string Image_path;
	cv::Ptr<cv::ml::KNearest> model;
};