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
#include "Knn_Recognition.h"

Knn_Recognition::Knn_Recognition()
{

}

Knn_Recognition::Knn_Recognition(std::string Image_path)
{
	this->Image_path = Image_path;
}

Knn_Recognition::~Knn_Recognition()
{

}

void Knn_Recognition::Knn_Training()
{
	this->model = Train_KNN();
}

void Knn_Recognition::detect(cv::Mat img, cv::Ptr<cv::ml::KNearest> model)
{
	/*****************                  在图片中获取校园卡区域                ********************/
	cv::Mat Interest_Area = Get_Card_Area(img);
	imwrite("Card_Area.jpg", Interest_Area);

	/*****************                  现在提取校园卡卡号区域                ********************/
	cv::Mat Num_Area = Get_Num_Area(Interest_Area);
	cv::imwrite("Num_area.jpg", Num_Area);

	/*****************                  对卡号区域进行分割识别                ********************/
	int *ans_array;
	ans_array = (int *)malloc(sizeof(int) * 10);
	ans_array = Identification_number(Num_Area, model);
	for (int i = 0; i <= 9; i++)
		std::cout << ans_array[i];
	std::cout << std::endl;
}

void Knn_Recognition::Solve()
{
	cv::namedWindow("img", 0);
	cv::Mat img = cv::imread(Image_path);
	imshow("img", img);
	Knn_Training();
	detect(img, model);
}

