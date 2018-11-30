#pragma once
#ifndef _Processing_function_H_
#define _Processing_function_H_

#include "opencv2\opencv.hpp"

/*
根据色调对卡片进行寻找，因为校园卡的色调都比较小
设置参数找到最佳的二值化阈值
*/
void  largestRect(cv::Mat img, cv::Mat &result);

/**
二值化操作，根据颜色进行二值化，只选亮度较低的部分
@param inputImage 校园卡区域
*/
cv::Mat  colorFilter(cv::Mat inputImage);

/**
去除光照影响
@param image 输入的原图像，灰度图或者三通道的原图输入都可以，输出图片一定为单通道灰度图
@param blockSize 推荐使用的大小为32
*/
void unevenLightCompensate(cv::Mat &image, int blockSize);

/**
使用mser方法对文字区域进行检测
@return 返回二值图，文字区域为白色
*/
cv::Mat Binarization(cv::Mat srcImage);

/**
@brief SauvolaThresh二值算法
此代码不适用与分辨率较大的图像

@param src 单通道灰度图
@param dst 单通道处理后的图
@param k  threshold = mean*(1 + k*((std / 128) - 1))
@param wndSize 处理区域宽高, 一定是奇数
*/
void SauvolaThresh(const cv::Mat src, cv::Mat& dst, const double k, int windowSize);

/**
@brief 在图片中得到校园卡区域
函数并没有完善好，因为旋转的情况还没有考虑

@param 整张图片
@return 返回的校园卡区域
*/
cv::Mat Get_Card_Area(cv::Mat img);

/**
@brief 根据校园卡区域得到校园卡卡号区域

@param Interest_Area 校园卡区域
@return 校园卡卡号区域
*/
cv::Mat Get_Num_Area(cv::Mat Interest_Area);

/**
数字识别
@param 校园卡卡号区域
*/
void Identification_number(cv::Mat Num_Area, cv::Ptr<cv::ml::KNearest> model);


#endif


