#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2\opencv.hpp"
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <algorithm>

#include "Processing_function.h"
#include "knn.h"

/*
根据色调对卡片进行寻找，因为校园卡的色调都比较小
设置参数找到最佳的二值化阈值
*/
void  largestRect(cv::Mat img, cv::Mat &result) {
	cv::Mat hsv;
	cvtColor(img, hsv, cv::COLOR_BGR2HSV);
	unevenLightCompensate(img, 32);
	std::vector<cv::Mat> channels;
	split(hsv, channels);
	std::vector<cv::Point> rect;

	double maxArea = 0.0;
	for (int i = 0; i <= 90; i++)
	{
		cv::Mat binaryImage = channels[1] < i;
		//imshow("thresh", binaryImage);

		if (countNonZero(binaryImage) > img.rows*img.cols*0.23)
		{
			result = binaryImage.clone();
			return;
		}
	}
}
/*
二值化操作，根据颜色进行二值化，只选亮度较低的部分,也就是黑色的部分
*/
cv::Mat  colorFilter(cv::Mat inputImage)
{
	cv::Mat image = inputImage.clone(), hsv;
	cv::Mat outputImage = cv::Mat(inputImage.size(), inputImage.type());
	cvtColor(image, hsv, cv::COLOR_BGR2HSV);
	int width = hsv.cols;
	int height = hsv.rows;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			if (!(hsv.at<cv::Vec3b>(cv::Point(j, i))[2] <= 90))        //非黑色部分
			{
				outputImage.at<cv::Vec3b>(cv::Point(j, i))[0] = 0;
				outputImage.at<cv::Vec3b>(cv::Point(j, i))[1] = 0;
				outputImage.at<cv::Vec3b>(cv::Point(j, i))[2] = 0;
			}
			else
			{
				outputImage.at<cv::Vec3b>(cv::Point(j, i))[0] = 255;
				outputImage.at<cv::Vec3b>(cv::Point(j, i))[1] = 255;
				outputImage.at<cv::Vec3b>(cv::Point(j, i))[2] = 255;
			}
		}
	imwrite("Binarization.jpg", outputImage);                      //二值化图片
	return outputImage;
}
/*
基本去除图片中因为阴影或者光照不均匀的影响
@param image 输入的原图像，灰度图或者三通道的原图输入都可以，输出图片一定为单通道灰度图
@param blockSize 推荐使用的大小为32
*/
void unevenLightCompensate(cv::Mat &image, int blockSize)
{
	if (image.channels() == 3)
		cvtColor(image, image, 7);
	double average = mean(image)[0];
	int rows_new = ceil(double(image.rows) / double(blockSize));
	int cols_new = ceil(double(image.cols) / double(blockSize));
	cv::Mat blockImage;
	blockImage = cv::Mat::zeros(rows_new, cols_new, CV_32FC1);
	for (int i = 0; i < rows_new; i++)
	{
		for (int j = 0; j < cols_new; j++)
		{
			int rowmin = i * blockSize;
			int rowmax = (i + 1)*blockSize;
			if (rowmax > image.rows) rowmax = image.rows;
			int colmin = j * blockSize;
			int colmax = (j + 1)*blockSize;
			if (colmax > image.cols) colmax = image.cols;
			cv::Mat imageROI = image(cv::Range(rowmin, rowmax), cv::Range(colmin, colmax));
			double temaver = mean(imageROI)[0];
			blockImage.at<float>(i, j) = temaver;
		}
	}
	blockImage = blockImage - average;
	cv::Mat blockImage2;
	resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), cv::INTER_CUBIC);
	cv::Mat image2;
	image.convertTo(image2, CV_32FC1);
	cv::Mat dst = image2 - blockImage2;
	dst.convertTo(image, CV_8UC1);
}
/**
利用mser+算法检测文字区域
*/
cv::Mat Binarization(cv::Mat srcImage)
{
	cv::Mat gray, gray_neg;
	cv::cvtColor(srcImage, gray, CV_BGR2HSV);
	// 灰度转换 
	cv::cvtColor(srcImage, gray, CV_BGR2GRAY);
	// 取反值灰度
	gray_neg = 255 - gray;
	std::vector<std::vector<cv::Point> > regContours;
	std::vector<std::vector<cv::Point> > charContours;

	// 创建MSER对象
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 10, 5000, 0.5, 0.3);
	cv::Ptr<cv::MSER> mesr2 = cv::MSER::create(2, 2, 400, 0.1, 0.3);


	std::vector<cv::Rect> bboxes1;
	std::vector<cv::Rect> bboxes2;
	// MSER+ 检测
	mesr1->detectRegions(gray, regContours, bboxes1);
	// MSER-操作
	mesr2->detectRegions(gray_neg, charContours, bboxes2);

	cv::Mat mserMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);
	cv::Mat mserNegMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);

	for (int i = (int)regContours.size() - 1; i >= 0; i--)
	{
		// 根据检测区域点生成mser+结果
		const std::vector<cv::Point>& r = regContours[i];
		for (int j = 0; j < (int)r.size(); j++)
		{
			cv::Point pt = r[j];
			mserMapMat.at<unsigned char>(pt) = 255;
		}
	}
	// MSER- 检测
	for (int i = (int)charContours.size() - 1; i >= 0; i--)
	{
		// 根据检测区域点生成mser-结果
		const std::vector<cv::Point>& r = charContours[i];
		for (int j = 0; j < (int)r.size(); j++)
		{
			cv::Point pt = r[j];
			mserNegMapMat.at<unsigned char>(pt) = 255;
		}
	}
	// mser结果输出
	cv::Mat mserResMat;
	// mser+与mser-位与操作
	mserResMat = mserMapMat & mserNegMapMat;
	//imwrite("mserMapMat.jpg", mserMapMat);
	//imwrite("mserNegMapMat.jpg", mserNegMapMat);
	//imwrite("mserResMat.jpg", mserResMat);
	return mserResMat;
}
static int CalcMaxValue(int a, int b)
{
	return (a > b) ? a : b;
}

static double CalcMaxValue(double a, double b)
{
	return (a > b) ? a : b;
}

static int CalcMinValue(int a, int b)
{
	return (a < b) ? a : b;
}

static double CalcMinValue(double a, double b)
{
	return (a < b) ? a : b;
}
/**
@brief SauvolaThresh二值算法
此代码不适用与分辨率较大的图像

@param src 单通道灰度图
@param dst 单通道处理后的图
@param k  threshold = mean*(1 + k*((std / 128) - 1))
@param wndSize 处理区域宽高, 一定是奇数
*/
void SauvolaThresh(const cv::Mat src, cv::Mat& dst, const double k, int windowSize)
{
	int whalf = windowSize >> 1;

	if (src.channels() == 3) 
		cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
	dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

	// 产生标志位图像
	unsigned long* integralImg = new unsigned long[src.rows * src.cols];
	unsigned long* integralImgSqrt = new unsigned long[src.rows * src.cols];
	std::memset(integralImg, 0, src.rows *src.cols * sizeof(unsigned long));
	std::memset(integralImgSqrt, 0, src.rows *src.cols * sizeof(unsigned long));

	// 计算直方图和图像值平方的和
	for (int y = 0; y < src.rows; ++y)
	{
		unsigned long sum = 0;
		unsigned long sqrtSum = 0;
		for (int x = 0; x < src.cols; ++x)
		{
			int index = y * src.cols + x;
			sum += src.at<uchar>(y, x);
			sqrtSum += src.at<uchar>(y, x)*src.at<uchar>(y, x);
			if (y == 0)
			{
				integralImg[index] = sum;
				integralImgSqrt[index] = sqrtSum;
			}
			else
			{
				integralImgSqrt[index] = integralImgSqrt[(y - 1)*src.cols + x] + sqrtSum;
				integralImg[index] = integralImg[(y - 1)*src.cols + x] + sum;
			}
		}
	}

	double diff = 0.0;
	double sqDiff = 0.0;
	double diagSum = 0.0;
	double iDiagSum = 0.0;
	double sqDiagSum = 0.0;
	double sqIDiagSum = 0.0;
	for (int x = 0; x < src.cols; ++x)
	{
		for (int y = 0; y < src.rows; ++y)
		{
			int xMin = CalcMaxValue(0, x - whalf);
			int yMin = CalcMaxValue(0, y - whalf);
			int xMax = CalcMinValue(src.cols - 1, x + whalf);
			int yMax = CalcMinValue(src.rows - 1, y + whalf);
			//std::cout << " xMax:" << xMax << " xMin:" << xMin << " yMax:" << yMax << " yMin" << yMin << std::endl;
			double area = (xMax - xMin + 1)*(yMax - yMin + 1);
			//std::cout << "area:" << area << std::endl;
			if (area <= 0)
			{
				dst.at<uchar>(y, x) = 255;
				continue;
			}

			if (xMin == 0 && yMin == 0)
			{
				diff = integralImg[yMax*src.cols + xMax];
				sqDiff = integralImgSqrt[yMax*src.cols + xMax];
			}
			else if (xMin > 0 && yMin == 0)
			{
				diff = integralImg[yMax*src.cols + xMax] - integralImg[yMax*src.cols + xMin - 1];
				sqDiff = integralImgSqrt[yMax * src.cols + xMax] - integralImgSqrt[yMax * src.cols + xMin - 1];
			}
			else if (xMin == 0 && yMin > 0)
			{
				diff = integralImg[yMax * src.cols + xMax] - integralImg[(yMin - 1) * src.cols + xMax];
				sqDiff = integralImgSqrt[yMax * src.cols + xMax] - integralImgSqrt[(yMin - 1) * src.cols + xMax];;
			}
			else
			{
				diagSum = integralImg[yMax * src.cols + xMax] + integralImg[(yMin - 1) * src.cols + xMin - 1];
				iDiagSum = integralImg[(yMin - 1) * src.cols + xMax] + integralImg[yMax * src.cols + xMin - 1];
				diff = diagSum - iDiagSum;
				sqDiagSum = integralImgSqrt[yMax * src.cols + xMax] + integralImgSqrt[(yMin - 1) * src.cols + xMin - 1];
				sqIDiagSum = integralImgSqrt[(yMin - 1) * src.cols + xMax] + integralImgSqrt[yMax * src.cols + xMin - 1];
				sqDiff = sqDiagSum - sqIDiagSum;
			}
			double mean = diff / area;
			double stdValue = sqrt((sqDiff - diff * diff / area) / (area - 1));
			double threshold = mean * (1 + k * ((stdValue / 128) - 1));
			//std::cout << threshold << " " << stdValue << std::endl;
			if (src.at<uchar>(y, x) < threshold)
				dst.at<uchar>(y, x) = 0;
			else
				dst.at<uchar>(y, x) = 255;
		}
	}
	delete[] integralImg;
	delete[] integralImgSqrt;
}

/*
在图片中得到校园卡区域,并返回
函数并没有完善好，因为旋转的情况还没有考虑
*/
cv::Mat Get_Card_Area(cv::Mat img)
{
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat binaryImage;
	cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2), 0, 0, cv::INTER_LINEAR);     //因为图片太大，所以要缩小一下，这里先缩小一半，后面要根据图片大小进行修改
	cv::Mat  img_ = img.clone();

	cv::GaussianBlur(img_, img_, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);
	///校园卡区域框选操作，使用hsv颜色模块中的色调进行筛选
	largestRect(img_, binaryImage);

	cv::Mat elemect = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));
	cv::dilate(binaryImage, binaryImage, elemect);


	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//CV_RETR_EXTERNAL只检测最外围轮廓，CV_CHAIN_APPROX_SIMPLE仅保存轮廓的拐点信息

	// 多边形近似
	int max_Area_position;
	double max_Area = 0;
	for (int i = 0; i < contours.size(); ++i)
	{
		std::vector<cv::Point> polygon;
		approxPolyDP(contours[i], polygon, arcLength(contours[i], 1) * 0.02, 1);
		double area = fabs(contourArea(polygon));
		if (max_Area < area)
		{
			max_Area = area;
			max_Area_position = i;
		}
	}
	//找到最大的矩形，基本上可以确定就是校园卡区域
	cv::RotatedRect rect = cv::minAreaRect(contours[max_Area_position]);
	cv::Point2f vertex[4];
	rect.points(vertex);

	imwrite("binaryImage.jpg", binaryImage);
	/*
	for (int size_i = 0; size_i < 4; size_i++)
	line(img, vertex[size_i], vertex[(size_i + 1) % 4], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

	imshow("img", img);
	*/

	///校园卡卡号区域提取节点（先忽略旋转等问题，这里的旋转也很好弄，之前项目中的旋转程序直接套取即可，就加在裁剪校园卡区域之前）

	cv::Mat Interest_Area;          //裁剪出来的校园卡区域
	cv::Rect s = cv::boundingRect(contours[max_Area_position]);
	Interest_Area = img_(cv::Rect(s.tl().x, s.tl().y, s.width, s.height));

	return Interest_Area;
}


/*
根据校园卡区域得到校园卡卡号区域
一共四种方案，下面有详细阐述，现在选用的是方案二
*/
cv::Mat Get_Num_Area(cv::Mat Interest_Area)
{
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat Interest_Area_copy = Interest_Area.clone();
	cv::Mat Interest_Area_gray, Interest_Area_erode, Interest_Area_dilate;

	//方案一、将蓝色通道去除，进行二值化处理                                                                       
	//vector<cv::Mat> channels; 
	//vector<cv::Mat> mbgr(3);   
	//cv::split(Interest_Area_copy, channels);
	//cv::Mat hideChannel(Interest_Area_copy.size(), CV_8UC1, cv::Scalar(0));
	//cv::Mat imageRG(Interest_Area_copy.size(), CV_8UC3);
	//mbgr[0] = hideChannel;	
	//mbgr[1] = channels[1];	
	//mbgr[2] = channels[2];	
	//merge(mbgr, imageRG);	
	//imshow("红色和绿色混合，无蓝", imageRG);
	//Interest_Area_copy = imageRG;


	//方案二、只取黑色区域
	cv::Mat Interest_Area_bin = colorFilter(Interest_Area_copy);
	cvtColor(Interest_Area_bin, Interest_Area_gray, cv::COLOR_BGR2GRAY);
	threshold(Interest_Area_gray, Interest_Area_bin, 0, 255, cv::THRESH_OTSU);

	//方案三、OTSU算法直接二值化
	//unevenLightCompensate(Interest_Area_copy, 32);
	//cout << Interest_Area_copy.type() << endl;
	//threshold(Interest_Area_copy, OTSU_img, 0, 255, cv::THRESH_OTSU);
	//OTSU_img = 255 - OTSU_img;
	//imwrite("OTSU_img.jpg", OTSU_img);
	//Interest_Area_copy = OTSU_img.clone();

	//方案四、smer算法二值化，需要时间较长一些
	//Interest_Area_copy = Binarization(Interest_Area_copy);
	//imwrite("threshold.jpg", Interest_Area_copy);

	//腐蚀膨胀操作获得连通域
	cv::Mat elemect_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(Interest_Area_copy.cols / 30, 3));
	cv::Mat elemect_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 2));

	cv::erode(Interest_Area_bin, Interest_Area_erode, elemect_erode, cv::Point(-1, -1));                      //腐蚀
	imwrite("Interest_Area_erode.jpg", Interest_Area_erode);
	cv::dilate(Interest_Area_erode, Interest_Area_dilate, elemect_dilate, cv::Point(-1, -1));                   //膨胀
	imwrite("Interest_Area_dilate.jpg", Interest_Area_dilate);

	//划取轮廓判断卡号区域
	double max_y = 0;
	int max_y_position = 0;
	findContours(Interest_Area_dilate, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	for (int size_i = 0; size_i < contours.size(); size_i++)
	{
		double area = cv::contourArea(contours[size_i]);

		if (area < 1000)
			continue;
		//找到最小外接矩形
		cv::RotatedRect rect = cv::minAreaRect(contours[size_i]);

		int m_width = rect.boundingRect().width;
		int m_height = rect.boundingRect().height;

		if (m_height > m_width * 0.25 || m_height < Interest_Area.rows / 25 || m_width > Interest_Area.cols / 4 * 3)
			continue;

		if (rect.boundingRect().tl().y > max_y && rect.boundingRect().tl().x < Interest_Area.cols / 2)
		{
			max_y = rect.boundingRect().tl().y;
			max_y_position = size_i;
		}
	}
	//assert(max_y != 0 && max_y_position != 0);
	/*
	rect = cv::minAreaRect(contours[max_y_position]);
	cv::Point2f P[4];
	rect.points(P);
	for (int j = 0; j <= 3; j++)
	line(Interest_Area, P[j], P[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);

	imwrite("img.jpg", Interest_Area);
	*/
	cv::Rect s = cv::boundingRect(contours[max_y_position]);
	cv::Mat Num_Area = Interest_Area(cv::Rect(s.tl().x, s.tl().y, s.width, s.height));
	cv::Mat Num_Area_bin = Interest_Area_bin(cv::Rect(s.tl().x, s.tl().y, s.width, s.height));          //色调H对图片进行的二值化操作

	//imshow("Num_Area.jpg", Num_Area);
	//imshow("Num_Area_bin.jpg", Num_Area_bin);
	return Num_Area;
}


/*
用来存储每个矩形的左上角的坐标，进行排序，然后我们选取前面十个作为学号区域
*/
typedef struct Num_Rect_
{
	double x, y;
	int order;

	bool operator <(Num_Rect_  &a)
	{
		if (x > a.x) return true;
		return false;
	}
}Num_Rect;
int* Identification_number(cv::Mat Num_Area, cv::Ptr<cv::ml::KNearest> model)
{
	std::vector<std::vector<cv::Point> > contours;
	cv::Mat binary,Num_Area_gray;
	cv::cvtColor(Num_Area, Num_Area_gray, cv::COLOR_BGR2GRAY);
	SauvolaThresh(Num_Area_gray, binary, 0.3, 20);
	binary = 255 - binary;
	cv::imwrite("Num_area_bin.jpg", binary);

	cv::copyMakeBorder(binary, binary, 5, 5, 5, 5, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));    //扩充图像边缘
	//imshow("binary", binary);

	findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat res = binary.clone();
	//std::cout << contours.size() << std::endl;
	Num_Rect num_rect[128];
	int pos = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		//std::cout << cv::contourArea(contours[i]) << std::endl;
		if (cv::contourArea(contours[i]) <= 50)
			continue;
		cv::Rect rect = cv::boundingRect(contours[i]);

		if (rect.height < binary.rows/2)
			continue;

		//更新Num_Rect结构体
		num_rect[pos].x = rect.tl().x;
		num_rect[pos].y = rect.tl().y;
		num_rect[pos++].order = i;
		/*
		//得到四个点的坐标
		cv::Point2f vertex[4];
		vertex[0] = rect.tl();                                                              //矩阵左上角的点
		vertex[1].x = (float)rect.tl().x, vertex[1].y = (float)rect.br().y;                 //矩阵左下方的点
		vertex[2] = rect.br();                                                              //矩阵右下角的点
		vertex[3].x = (float)rect.br().x, vertex[3].y = (float)rect.tl().y;                 //矩阵右上方的点

		for (int j = 0; j < 4; j++)
			line(res, vertex[j], vertex[(j + 1) % 4], cv::Scalar(255, 255, 255), 1);
		*/
	}
	std::sort(num_rect, num_rect + pos);

	//for (int i = 0; i < pos; i++)
	//	std::cout << num_rect[i].x << "   " << num_rect[i].y << "  " << num_rect[i].order << std::endl;

	cv::Mat res_ = cv::Mat(res.size(), res.type());
	int num = 0;
	int ans_array[10];
	pos = std::min(pos, 10);
	assert(pos == 10);                                                                        //识别成功的的时候，pos一定等于10
	for (int i = pos - 1; i >= 0; i--)
	{
		if (num++ == 10) break;
		cv::Rect rect = cv::boundingRect(contours[num_rect[i].order]);

		//得到四个点的坐标
		cv::Point2f vertex[4];
		vertex[0] = rect.tl();                                                              //矩阵左上角的点
		vertex[1].x = (float)rect.tl().x, vertex[1].y = (float)rect.br().y;                 //矩阵左下方的点
		vertex[2] = rect.br();                                                              //矩阵右下角的点
		vertex[3].x = (float)rect.br().x, vertex[3].y = (float)rect.tl().y;                 //矩阵右上方的点

		cv::Mat part = res(cv::Rect(rect.tl().x, rect.tl().y, rect.width, rect.height));
		//cv::copyMakeBorder(part, part, 10, 10, 5, 5, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

		part.convertTo(part, CV_32F);
		cv::resize(part, part, cv::Size(17, 27));
		//imshow("part", part);
		int ans = Predict_KNN(model, part);
		//std::cout << ans << std::endl;
		ans_array[num - 1] = ans;
		//imwrite(std::to_string(num_rect[i].order) + ".jpg", part);
		//cv::waitKey(0);
	}
	for (int i = 0; i <= 9; i++)
		std::cout << ans_array[i];
	std::cout << std::endl;
	return ans_array;
}

