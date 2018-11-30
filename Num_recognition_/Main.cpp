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


/*
//����������
void on_Trackbar(int ,void *)
{
	//cv::Mat res;
	//Canny(img, res, min_threshold, max_threshold, 3);
	//imshow("cont", res);
	//cv::Mat thresh;
	//threshold(gray, thresh, min_threshold, 255, cv::THRESH_BINARY_INV);
	//imshow("thresh", thresh);
	//findContours(thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//for (int i = 0; i < contours.size(); i++)
	//	drawContours(cont, contours, i, Scalar(255, 255, 255));
	//imshow("img", cont);
}
*/

int main()
{
	Knn_Recognition knn_recognition;
	knn_recognition.Set_Knn_Recognition_Image_path("F:/ͼ��ʶ����Ŀ/У԰��ѧ��ʶ��/test_num_2.jpg");
	knn_recognition.Solve();

	//cv::Mat img;
	//cv::namedWindow("img", 0);
	//img = cv::imread("C:/Users/dell/Desktop/ͼ��ʶ����Ŀ/У԰��ѧ��ʶ��/test_num_7.jpg");
	//imshow("img", img);

	/*****************                     knn��������ѵ��                   ********************/
	//cv::Ptr<cv::ml::KNearest> model = Train_KNN();

	/*****************                  ��ͼƬ�л�ȡУ԰������                ********************/
	//cv::Mat Interest_Area = Get_Card_Area(img);
	//imwrite("Card_Area.jpg", Interest_Area);

	/*****************                  ������ȡУ԰����������                ********************/
	//cv::Mat Num_Area = Get_Num_Area(Interest_Area);
	//cv::imwrite("Num_area.jpg", Num_Area);

	/*****************                  �Կ���������зָ�ʶ��                ********************/
	//int *ans_array;
	//ans_array = (int *)malloc(sizeof(int) * 10);
	//ans_array = Identification_number(Num_Area, model);
	//for (int i = 0; i <= 9; i++)
	//	std::cout << ans_array[i];
	//std::cout << std::endl;

	cv::waitKey(0);

	/*
	//imshow("cont", img);
	//cont = cv::Mat::zeros(img.size(),CV_8UC1);
	//cv::createTrackbar("��ֵ1", "cont", &min_threshold, 255, on_Trackbar);
	//cv::createTrackbar("��ֵ2", "cont", &max_threshold, 255, on_Trackbar);

	//ʹ��Soble������ȡ��Ե��Ϣ��
	cv::Mat gray_x, gray_y;
	cv::Sobel(gray, gray_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(gray_x, gray_x);

	cv::Sobel(gray, gray_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(gray_y, gray_y);

	cv::addWeighted(gray_x, 0.5, gray_y, 0.5, 0, gray);
	imshow("cont", gray);
	
	/*
	//����任���ֱ��
	vector<cv::Vec4i> lines;

	HoughLinesP(gray, lines, 1, CV_PI / 180, 80, 50, 1000);

	for (int i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		line(img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
	}
	imshow("img", img);
	*/
	/*
	//����任���Բ
	vector<cv::Vec3f> circles;
	cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1.5, 200);
	for (int size_i = 0; size_i < circles.size(); size_i++)
	{
		cv::Point center(cvRound(circles[size_i][0]), cvRound(circles[size_i][1]));
		int radius = cvRound(circles[size_i][2]);

		cv::circle(gray, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		cv::circle(gray, center, radius, cv::Scalar(0, 255, 0), 3, 8, 0);
	}
	imshow("gray", gray);
	*/
	return 0;
}