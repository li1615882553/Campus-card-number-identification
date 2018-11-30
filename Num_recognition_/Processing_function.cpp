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
����ɫ���Կ�Ƭ����Ѱ�ң���ΪУ԰����ɫ�����Ƚ�С
���ò����ҵ���ѵĶ�ֵ����ֵ
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
��ֵ��������������ɫ���ж�ֵ����ֻѡ���Ƚϵ͵Ĳ���,Ҳ���Ǻ�ɫ�Ĳ���
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
			if (!(hsv.at<cv::Vec3b>(cv::Point(j, i))[2] <= 90))        //�Ǻ�ɫ����
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
	imwrite("Binarization.jpg", outputImage);                      //��ֵ��ͼƬ
	return outputImage;
}
/*
����ȥ��ͼƬ����Ϊ��Ӱ���߹��ղ����ȵ�Ӱ��
@param image �����ԭͼ�񣬻Ҷ�ͼ������ͨ����ԭͼ���붼���ԣ����ͼƬһ��Ϊ��ͨ���Ҷ�ͼ
@param blockSize �Ƽ�ʹ�õĴ�СΪ32
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
����mser+�㷨�����������
*/
cv::Mat Binarization(cv::Mat srcImage)
{
	cv::Mat gray, gray_neg;
	cv::cvtColor(srcImage, gray, CV_BGR2HSV);
	// �Ҷ�ת�� 
	cv::cvtColor(srcImage, gray, CV_BGR2GRAY);
	// ȡ��ֵ�Ҷ�
	gray_neg = 255 - gray;
	std::vector<std::vector<cv::Point> > regContours;
	std::vector<std::vector<cv::Point> > charContours;

	// ����MSER����
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 10, 5000, 0.5, 0.3);
	cv::Ptr<cv::MSER> mesr2 = cv::MSER::create(2, 2, 400, 0.1, 0.3);


	std::vector<cv::Rect> bboxes1;
	std::vector<cv::Rect> bboxes2;
	// MSER+ ���
	mesr1->detectRegions(gray, regContours, bboxes1);
	// MSER-����
	mesr2->detectRegions(gray_neg, charContours, bboxes2);

	cv::Mat mserMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);
	cv::Mat mserNegMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);

	for (int i = (int)regContours.size() - 1; i >= 0; i--)
	{
		// ���ݼ�����������mser+���
		const std::vector<cv::Point>& r = regContours[i];
		for (int j = 0; j < (int)r.size(); j++)
		{
			cv::Point pt = r[j];
			mserMapMat.at<unsigned char>(pt) = 255;
		}
	}
	// MSER- ���
	for (int i = (int)charContours.size() - 1; i >= 0; i--)
	{
		// ���ݼ�����������mser-���
		const std::vector<cv::Point>& r = charContours[i];
		for (int j = 0; j < (int)r.size(); j++)
		{
			cv::Point pt = r[j];
			mserNegMapMat.at<unsigned char>(pt) = 255;
		}
	}
	// mser������
	cv::Mat mserResMat;
	// mser+��mser-λ�����
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
@brief SauvolaThresh��ֵ�㷨
�˴��벻������ֱ��ʽϴ��ͼ��

@param src ��ͨ���Ҷ�ͼ
@param dst ��ͨ��������ͼ
@param k  threshold = mean*(1 + k*((std / 128) - 1))
@param wndSize ����������, һ��������
*/
void SauvolaThresh(const cv::Mat src, cv::Mat& dst, const double k, int windowSize)
{
	int whalf = windowSize >> 1;

	if (src.channels() == 3) 
		cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
	dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

	// ������־λͼ��
	unsigned long* integralImg = new unsigned long[src.rows * src.cols];
	unsigned long* integralImgSqrt = new unsigned long[src.rows * src.cols];
	std::memset(integralImg, 0, src.rows *src.cols * sizeof(unsigned long));
	std::memset(integralImgSqrt, 0, src.rows *src.cols * sizeof(unsigned long));

	// ����ֱ��ͼ��ͼ��ֵƽ���ĺ�
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
��ͼƬ�еõ�У԰������,������
������û�����ƺã���Ϊ��ת�������û�п���
*/
cv::Mat Get_Card_Area(cv::Mat img)
{
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat binaryImage;
	cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2), 0, 0, cv::INTER_LINEAR);     //��ΪͼƬ̫������Ҫ��Сһ�£���������Сһ�룬����Ҫ����ͼƬ��С�����޸�
	cv::Mat  img_ = img.clone();

	cv::GaussianBlur(img_, img_, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);
	///У԰�������ѡ������ʹ��hsv��ɫģ���е�ɫ������ɸѡ
	largestRect(img_, binaryImage);

	cv::Mat elemect = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));
	cv::dilate(binaryImage, binaryImage, elemect);


	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//CV_RETR_EXTERNALֻ�������Χ������CV_CHAIN_APPROX_SIMPLE�����������Ĺյ���Ϣ

	// ����ν���
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
	//�ҵ����ľ��Σ������Ͽ���ȷ������У԰������
	cv::RotatedRect rect = cv::minAreaRect(contours[max_Area_position]);
	cv::Point2f vertex[4];
	rect.points(vertex);

	imwrite("binaryImage.jpg", binaryImage);
	/*
	for (int size_i = 0; size_i < 4; size_i++)
	line(img, vertex[size_i], vertex[(size_i + 1) % 4], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

	imshow("img", img);
	*/

	///У԰������������ȡ�ڵ㣨�Ⱥ�����ת�����⣬�������תҲ�ܺ�Ū��֮ǰ��Ŀ�е���ת����ֱ����ȡ���ɣ��ͼ��ڲü�У԰������֮ǰ��

	cv::Mat Interest_Area;          //�ü�������У԰������
	cv::Rect s = cv::boundingRect(contours[max_Area_position]);
	Interest_Area = img_(cv::Rect(s.tl().x, s.tl().y, s.width, s.height));

	return Interest_Area;
}


/*
����У԰������õ�У԰����������
һ�����ַ�������������ϸ����������ѡ�õ��Ƿ�����
*/
cv::Mat Get_Num_Area(cv::Mat Interest_Area)
{
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat Interest_Area_copy = Interest_Area.clone();
	cv::Mat Interest_Area_gray, Interest_Area_erode, Interest_Area_dilate;

	//����һ������ɫͨ��ȥ�������ж�ֵ������                                                                       
	//vector<cv::Mat> channels; 
	//vector<cv::Mat> mbgr(3);   
	//cv::split(Interest_Area_copy, channels);
	//cv::Mat hideChannel(Interest_Area_copy.size(), CV_8UC1, cv::Scalar(0));
	//cv::Mat imageRG(Interest_Area_copy.size(), CV_8UC3);
	//mbgr[0] = hideChannel;	
	//mbgr[1] = channels[1];	
	//mbgr[2] = channels[2];	
	//merge(mbgr, imageRG);	
	//imshow("��ɫ����ɫ��ϣ�����", imageRG);
	//Interest_Area_copy = imageRG;


	//��������ֻȡ��ɫ����
	cv::Mat Interest_Area_bin = colorFilter(Interest_Area_copy);
	cvtColor(Interest_Area_bin, Interest_Area_gray, cv::COLOR_BGR2GRAY);
	threshold(Interest_Area_gray, Interest_Area_bin, 0, 255, cv::THRESH_OTSU);

	//��������OTSU�㷨ֱ�Ӷ�ֵ��
	//unevenLightCompensate(Interest_Area_copy, 32);
	//cout << Interest_Area_copy.type() << endl;
	//threshold(Interest_Area_copy, OTSU_img, 0, 255, cv::THRESH_OTSU);
	//OTSU_img = 255 - OTSU_img;
	//imwrite("OTSU_img.jpg", OTSU_img);
	//Interest_Area_copy = OTSU_img.clone();

	//�����ġ�smer�㷨��ֵ������Ҫʱ��ϳ�һЩ
	//Interest_Area_copy = Binarization(Interest_Area_copy);
	//imwrite("threshold.jpg", Interest_Area_copy);

	//��ʴ���Ͳ��������ͨ��
	cv::Mat elemect_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(Interest_Area_copy.cols / 30, 3));
	cv::Mat elemect_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 2));

	cv::erode(Interest_Area_bin, Interest_Area_erode, elemect_erode, cv::Point(-1, -1));                      //��ʴ
	imwrite("Interest_Area_erode.jpg", Interest_Area_erode);
	cv::dilate(Interest_Area_erode, Interest_Area_dilate, elemect_dilate, cv::Point(-1, -1));                   //����
	imwrite("Interest_Area_dilate.jpg", Interest_Area_dilate);

	//��ȡ�����жϿ�������
	double max_y = 0;
	int max_y_position = 0;
	findContours(Interest_Area_dilate, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	for (int size_i = 0; size_i < contours.size(); size_i++)
	{
		double area = cv::contourArea(contours[size_i]);

		if (area < 1000)
			continue;
		//�ҵ���С��Ӿ���
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
	cv::Mat Num_Area_bin = Interest_Area_bin(cv::Rect(s.tl().x, s.tl().y, s.width, s.height));          //ɫ��H��ͼƬ���еĶ�ֵ������

	//imshow("Num_Area.jpg", Num_Area);
	//imshow("Num_Area_bin.jpg", Num_Area_bin);
	return Num_Area;
}


/*
�����洢ÿ�����ε����Ͻǵ����꣬��������Ȼ������ѡȡǰ��ʮ����Ϊѧ������
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

	cv::copyMakeBorder(binary, binary, 5, 5, 5, 5, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));    //����ͼ���Ե
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

		//����Num_Rect�ṹ��
		num_rect[pos].x = rect.tl().x;
		num_rect[pos].y = rect.tl().y;
		num_rect[pos++].order = i;
		/*
		//�õ��ĸ��������
		cv::Point2f vertex[4];
		vertex[0] = rect.tl();                                                              //�������Ͻǵĵ�
		vertex[1].x = (float)rect.tl().x, vertex[1].y = (float)rect.br().y;                 //�������·��ĵ�
		vertex[2] = rect.br();                                                              //�������½ǵĵ�
		vertex[3].x = (float)rect.br().x, vertex[3].y = (float)rect.tl().y;                 //�������Ϸ��ĵ�

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
	assert(pos == 10);                                                                        //ʶ��ɹ��ĵ�ʱ��posһ������10
	for (int i = pos - 1; i >= 0; i--)
	{
		if (num++ == 10) break;
		cv::Rect rect = cv::boundingRect(contours[num_rect[i].order]);

		//�õ��ĸ��������
		cv::Point2f vertex[4];
		vertex[0] = rect.tl();                                                              //�������Ͻǵĵ�
		vertex[1].x = (float)rect.tl().x, vertex[1].y = (float)rect.br().y;                 //�������·��ĵ�
		vertex[2] = rect.br();                                                              //�������½ǵĵ�
		vertex[3].x = (float)rect.br().x, vertex[3].y = (float)rect.tl().y;                 //�������Ϸ��ĵ�

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

