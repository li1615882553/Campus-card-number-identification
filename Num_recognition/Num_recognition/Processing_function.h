#pragma once
#ifndef _Processing_function_H_
#define _Processing_function_H_

#include "opencv2\opencv.hpp"

/*
����ɫ���Կ�Ƭ����Ѱ�ң���ΪУ԰����ɫ�����Ƚ�С
���ò����ҵ���ѵĶ�ֵ����ֵ
*/
void  largestRect(cv::Mat img, cv::Mat &result);

/**
��ֵ��������������ɫ���ж�ֵ����ֻѡ���Ƚϵ͵Ĳ���
@param inputImage У԰������
*/
cv::Mat  colorFilter(cv::Mat inputImage);

/**
ȥ������Ӱ��
@param image �����ԭͼ�񣬻Ҷ�ͼ������ͨ����ԭͼ���붼���ԣ����ͼƬһ��Ϊ��ͨ���Ҷ�ͼ
@param blockSize �Ƽ�ʹ�õĴ�СΪ32
*/
void unevenLightCompensate(cv::Mat &image, int blockSize);

/**
ʹ��mser����������������м��
@return ���ض�ֵͼ����������Ϊ��ɫ
*/
cv::Mat Binarization(cv::Mat srcImage);

/**
@brief SauvolaThresh��ֵ�㷨
�˴��벻������ֱ��ʽϴ��ͼ��

@param src ��ͨ���Ҷ�ͼ
@param dst ��ͨ��������ͼ
@param k  threshold = mean*(1 + k*((std / 128) - 1))
@param wndSize ����������, һ��������
*/
void SauvolaThresh(const cv::Mat src, cv::Mat& dst, const double k, int windowSize);

/**
@brief ��ͼƬ�еõ�У԰������
������û�����ƺã���Ϊ��ת�������û�п���

@param ����ͼƬ
@return ���ص�У԰������
*/
cv::Mat Get_Card_Area(cv::Mat img);

/**
@brief ����У԰������õ�У԰����������

@param Interest_Area У԰������
@return У԰����������
*/
cv::Mat Get_Num_Area(cv::Mat Interest_Area);

/**
����ʶ��
@param У԰����������
*/
void Identification_number(cv::Mat Num_Area, cv::Ptr<cv::ml::KNearest> model);


#endif


