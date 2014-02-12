#include "BackgroundMaskCleaner.h"
#include <opencv2\opencv.hpp>

using namespace cv;

BackgroundMaskCleaner::BackgroundMaskCleaner()
{
	this->se21 = NULL;
	this->se11 = NULL;
}


BackgroundMaskCleaner::~BackgroundMaskCleaner()
{
	cvReleaseStructuringElement(&se21);
	cvReleaseStructuringElement(&se11);
}


void BackgroundMaskCleaner::cleanMask(cv::Mat src)
{
	/// init the structuring elements if they don't exist
	if (this->se21 == NULL)
		this->se21 = cvCreateStructuringElementEx(15, 15, 10, 10, CV_SHAPE_RECT, NULL); //21,10
	if (this->se11 == NULL)
		this->se11 = cvCreateStructuringElementEx(10, 10, 5, 5,  CV_SHAPE_RECT, NULL); //10,5

	// convert to the older OpenCV image format to use the algorithms below
	IplImage srcCvt = src;

	// run some morphs on the mask to get rid of noise
	cvMorphologyEx(&srcCvt, &srcCvt, 0, this->se11, CV_MOP_OPEN, 1);
	cvMorphologyEx(&srcCvt, &srcCvt, 0, this->se21, CV_MOP_CLOSE, 1);
}