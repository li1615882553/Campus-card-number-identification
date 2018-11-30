#include "opencv2\opencv.hpp"
#include <iostream>


cv::Ptr<cv::ml::KNearest> Train_KNN()
{
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat img = cv::imread("./digit.png");
	cvtColor(img, img, CV_BGR2GRAY);
	threshold(img, img, 0, 255, cv::THRESH_OTSU);
	cv::Mat data, labels, part;   //特征矩阵

	findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++)
	{
		cv::Rect rect = boundingRect(contours[i]);

		cv::Point2f vertex[4];
		vertex[0] = rect.tl();                                                              //矩阵左上角的点
		vertex[1].x = (float)rect.tl().x, vertex[1].y = (float)rect.br().y;                 //矩阵左下方的点
		vertex[2] = rect.br();                                                              //矩阵右下角的点
		vertex[3].x = (float)rect.br().x, vertex[3].y = (float)rect.tl().y;                 //矩阵右上方的点

		img(cv::Rect(rect.tl().x, rect.tl().y, rect.width, rect.height)).copyTo(part);         // 大小：17*27

		cv::resize(part, part, cv::Size(17, 27));

		data.push_back(part.reshape(0, 1));
		labels.push_back((int)(-1 * ((i) / 20) + 9));
	}

	/*
	int b = 20;
	int m = gray.rows / b;   //原图为1000*2000
	int n = gray.cols / b;   //裁剪为5000个20*20的小图块

	for (int i = 0; i < n; i++)
	{
	int offsetCol = i * b; //列上的偏移量,因为一个图片大小为20*20
	for (int j = 0; j < m; j++)
	{
	int offsetRow = j * b;  //行上的偏移量
	//截取20*20的小块
	cv::Mat tmp;
	gray(cv::Range(offsetRow, offsetRow + b), cv::Range(offsetCol, offsetCol + b)).copyTo(tmp);
	data.push_back(tmp.reshape(0, 1));  //序列化后放入特征矩阵,也就是将矩阵转换为一行，然后放入data中进行储存
	labels.push_back((int)j / 5);  //对应的识别结果
	}
	}
	*/



	data.convertTo(data, CV_32F); //uchar型转换为cv_32f
	int samplesNum = data.rows;
	int trainNum = 200;
	cv::Mat trainData, trainLabels;
	trainData = data(cv::Range(0, trainNum), cv::Range::all());
	trainLabels = labels(cv::Range(0, trainNum), cv::Range::all());

	//使用KNN算法
	int K = 7;
	cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLabels);
	cv::Ptr<cv::ml::KNearest> model = cv::ml::KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tData);
	return model;
}

float  Predict_KNN(cv::Ptr<cv::ml::KNearest> model, cv::Mat sample)
{
	sample = sample.reshape(0, 1);
	float forecast_result = model->predict(sample);   //对所有行进行预测
	return forecast_result;
}