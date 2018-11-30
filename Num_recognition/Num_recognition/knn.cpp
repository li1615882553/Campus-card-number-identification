#include "opencv2\opencv.hpp"
#include <iostream>


cv::Ptr<cv::ml::KNearest> Train_KNN()
{
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat img = cv::imread("./digit.png");
	cvtColor(img, img, CV_BGR2GRAY);
	threshold(img, img, 0, 255, cv::THRESH_OTSU);
	cv::Mat data, labels, part;   //��������

	findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++)
	{
		cv::Rect rect = boundingRect(contours[i]);

		cv::Point2f vertex[4];
		vertex[0] = rect.tl();                                                              //�������Ͻǵĵ�
		vertex[1].x = (float)rect.tl().x, vertex[1].y = (float)rect.br().y;                 //�������·��ĵ�
		vertex[2] = rect.br();                                                              //�������½ǵĵ�
		vertex[3].x = (float)rect.br().x, vertex[3].y = (float)rect.tl().y;                 //�������Ϸ��ĵ�

		img(cv::Rect(rect.tl().x, rect.tl().y, rect.width, rect.height)).copyTo(part);         // ��С��17*27

		cv::resize(part, part, cv::Size(17, 27));

		data.push_back(part.reshape(0, 1));
		labels.push_back((int)(-1 * ((i) / 20) + 9));
	}

	/*
	int b = 20;
	int m = gray.rows / b;   //ԭͼΪ1000*2000
	int n = gray.cols / b;   //�ü�Ϊ5000��20*20��Сͼ��

	for (int i = 0; i < n; i++)
	{
	int offsetCol = i * b; //���ϵ�ƫ����,��Ϊһ��ͼƬ��СΪ20*20
	for (int j = 0; j < m; j++)
	{
	int offsetRow = j * b;  //���ϵ�ƫ����
	//��ȡ20*20��С��
	cv::Mat tmp;
	gray(cv::Range(offsetRow, offsetRow + b), cv::Range(offsetCol, offsetCol + b)).copyTo(tmp);
	data.push_back(tmp.reshape(0, 1));  //���л��������������,Ҳ���ǽ�����ת��Ϊһ�У�Ȼ�����data�н��д���
	labels.push_back((int)j / 5);  //��Ӧ��ʶ����
	}
	}
	*/



	data.convertTo(data, CV_32F); //uchar��ת��Ϊcv_32f
	int samplesNum = data.rows;
	int trainNum = 200;
	cv::Mat trainData, trainLabels;
	trainData = data(cv::Range(0, trainNum), cv::Range::all());
	trainLabels = labels(cv::Range(0, trainNum), cv::Range::all());

	//ʹ��KNN�㷨
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
	float forecast_result = model->predict(sample);   //�������н���Ԥ��
	return forecast_result;
}