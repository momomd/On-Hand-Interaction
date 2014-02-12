#include <stdio.h>
#include <conio.h>
#include <windows.h>
#include <math.h>
#include <wchar.h>
#include <vector>
#include <tchar.h>
#include <cmath>
#include "stdafx.h"
#include <string.h>
#include <fstream>
#include <iostream>
#include "pxcsession.h"
#include "pxcsmartptr.h"
#include "pxccapture.h"
#include "util_render.h"
#include "util_capture_file.h"
#include "util_cmdline.h"
#include "util_pipeline.h"
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2/legacy/legacy.hpp>"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/flann/flann.hpp>
#include "CaptureStream.h"
#include "BackgroundMaskCleaner.h"


using namespace std;
using namespace cv;

	int Thresholdness = 50;
	int ialpha = 10;
int ibeta=10; 
int igamma=10; 

int imageNum=0;

Mat depthFrame;
Mat binary,interactM,testM,edgeM;
Mat refH,pntH;
int leftI, leftJ, leftX, topY,topX,bottomY;
std::vector<std::vector<Point>> contoursL;
CvSeq* contourSeqL = 0;
RotatedRect ellipseL, ellipseR, erodeL;
std::vector<std::vector<Point>> contoursR;
RNG rng(12345);
Rect handFrame;
bool touched = false;
Point touchPnt;
bool showContour = true;
int mode = 0; 
IplImage *image = 0 ; 
IplImage *image2 = 0 ; 

cv::Mat falseColorsMap;

Mat imageM; 
Mat imageM2; 


IplImage* imgTracking;
int lastX = -1;
int lastY = -1;
Rect redR;

#define THRESHOLD 150
#define BRIGHT 0.7
#define DARK 0.2


static void draw_subdiv_point( Mat& img, Point2f fp, Scalar color )
{
	circle( img, fp, 3, color, -1, 8, 0 );
}

static void draw_subdivPre( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color )
{

	vector<Vec4f> edgeList;
	subdiv.getEdgeList(edgeList);
	for( size_t i = 0; i < edgeList.size(); i++ )
	{
		Vec4f e = edgeList[i];
		Point pt0 = Point(cvRound(e[0]), cvRound(e[1]));
		Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));
		line(img, pt0, pt1, delaunay_color, 1, 8, 0);
	}

}

static void draw_subdiv( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color )
{
#if 1
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);

	for( size_t i = 0; i < triangleList.size(); i++ )
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt[0], pt[1], delaunay_color, 1, 8, 0);
		line(img, pt[1], pt[2], delaunay_color, 1, 8, 0);
		line(img, pt[2], pt[0], delaunay_color, 1, 8, 0);

	}
#else
	vector<Vec4f> edgeList;
	subdiv.getEdgeList(edgeList);
	for( size_t i = 0; i < edgeList.size(); i++ )
	{
		Vec4f e = edgeList[i];
		Point pt0 = Point(cvRound(e[0]), cvRound(e[1]));
		Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));
		line(img, pt0, pt1, delaunay_color, 1, 8, 0);
	}
#endif
}

static void locate_point( Mat& img, Subdiv2D& subdiv, Point2f fp, Scalar active_color )
{
	int e0=0, vertex=0;

	subdiv.locate(fp, e0, vertex);

	if( e0 > 0 )
	{
		int e = e0;
		do
		{
			Point2f org, dst;
			if( subdiv.edgeOrg(e, &org) > 0 && subdiv.edgeDst(e, &dst) > 0 )
				line( img, org, dst, active_color, 3, 8, 0 );

			e = subdiv.getEdge(e, Subdiv2D::NEXT_AROUND_LEFT);
		}
		while( e != e0 );
	}

	draw_subdiv_point( img, fp, active_color );
}


static void paint_voronoi( Mat& img, Subdiv2D& subdiv )
{
	vector<vector<Point2f> > facets;
	vector<Point2f> centers;
	subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

	vector<Point> ifacet;
	vector<vector<Point> > ifacets(1);

	for( size_t i = 0; i < facets.size(); i++ )
	{
		ifacet.resize(facets[i].size());
		for( size_t j = 0; j < facets[i].size(); j++ )
			ifacet[j] = facets[i][j];

		Scalar color;
		color[0] = rand() & 255;
		color[1] = rand() & 255;
		color[2] = rand() & 255;
		fillConvexPoly(img, ifacet, color, 8, 0);

		ifacets[0] = ifacet;
		polylines(img, ifacets, true, Scalar(), 1, 8, 0);
		circle(img, centers[i], 3, Scalar(), -1, 8, 0);
	}
}

//scale, rotation invarient template matching
void TemplateMatch() //takes too long
{

	int i, j, x, y, key;
	double minVal;
	char windowNameSource[] = "Original Image";
	char windowNameDestination[] = "Result Image";
	char windowNameCoefficientOfCorrelation[] = "Coefficient of Correlation Image";
	CvPoint minLoc;
	CvPoint tempLoc;

	IplImage *sourceImage = cvLoadImage("template_source.jpg", CV_LOAD_IMAGE_ANYDEPTH         | CV_LOAD_IMAGE_ANYCOLOR);
	IplImage *templateImage = cvLoadImage("template.jpg", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);



	IplImage *graySourceImage = cvCreateImage(cvGetSize(sourceImage), IPL_DEPTH_8U, 1); 
	IplImage *grayTemplateImage =cvCreateImage(cvGetSize(templateImage),IPL_DEPTH_8U,1);
	IplImage *binarySourceImage = cvCreateImage(cvGetSize(sourceImage), IPL_DEPTH_8U, 1); 
	IplImage *binaryTemplateImage = cvCreateImage(cvGetSize(templateImage), IPL_DEPTH_8U, 1); 
	IplImage *destinationImage = cvCreateImage(cvGetSize(sourceImage), IPL_DEPTH_8U, 3); 

	cvCopy(sourceImage, destinationImage);

	cvCvtColor(sourceImage, graySourceImage, CV_RGB2GRAY);
	cvCvtColor(templateImage, grayTemplateImage, CV_RGB2GRAY);

	cvThreshold(graySourceImage, binarySourceImage, 200, 255, CV_THRESH_OTSU );
	cvThreshold(grayTemplateImage, binaryTemplateImage, 200, 255, CV_THRESH_OTSU);

	int templateHeight = templateImage->height;
	int templateWidth = templateImage->width;

	float templateScale = 0.5f;

	for(i = 2; i <= 3; i++) 
	{

		int tempTemplateHeight = (int)(templateWidth * (i * templateScale));
		int tempTemplateWidth = (int)(templateHeight * (i * templateScale));

		IplImage *tempBinaryTemplateImage = cvCreateImage(cvSize(tempTemplateWidth,                  tempTemplateHeight), IPL_DEPTH_8U, 1);

		// W - w + 1, H - h + 1

		IplImage *result = cvCreateImage(cvSize(sourceImage->width - tempTemplateWidth + 1,      sourceImage->height - tempTemplateHeight + 1), IPL_DEPTH_32F, 1);

		cvResize(binaryTemplateImage, tempBinaryTemplateImage, CV_INTER_LINEAR);
		float degree = 20.0f;
		for(j = 0; j <= 9; j++) 
		{

			IplImage *rotateBinaryTemplateImage = cvCreateImage(cvSize(tempBinaryTemplateImage->width, tempBinaryTemplateImage->height), IPL_DEPTH_8U, 1);

			//cvShowImage(windowNameSource, tempBinaryTemplateImage);  
			//cvWaitKey(0);             

			for(y = 0; y < tempTemplateHeight; y++)
			{

				for(x = 0; x < tempTemplateWidth; x++)
				{
					rotateBinaryTemplateImage->imageData[y * tempTemplateWidth + x] = 255;

				}         
			}


			for(y = 0; y < tempTemplateHeight; y++)
			{

				for(x = 0; x < tempTemplateWidth; x++)
				{

					float radian = (float)j * degree * CV_PI / 180.0f;
					int scale = y * tempTemplateWidth + x;

					int rotateY = - sin(radian) * ((float)x - (float)tempTemplateWidth / 2.0f) + cos(radian) * ((float)y - (float)tempTemplateHeight / 2.0f) + tempTemplateHeight / 2;

					int rotateX = cos(radian) * ((float)x - (float)tempTemplateWidth / 2.0f) + sin(radian) * ((float)y - (float)tempTemplateHeight / 2.0f) + tempTemplateWidth / 2;


					if(rotateY < tempTemplateHeight && rotateX < tempTemplateWidth && rotateY >= 0 && rotateX  >= 0)

						rotateBinaryTemplateImage->imageData[scale] = tempBinaryTemplateImage->imageData[rotateY * tempTemplateWidth + rotateX];

				}

			}


			//cvShowImage(windowNameSource, rotateBinaryTemplateImage);
			//cvWaitKey(0);

			cvMatchTemplate(binarySourceImage, rotateBinaryTemplateImage, result, CV_TM_SQDIFF_NORMED); 

			//cvMatchTemplate(binarySourceImage, rotateBinaryTemplateImage, result, CV_TM_SQDIFF);  

			cvMinMaxLoc(result, &minVal, NULL, &minLoc, NULL, NULL);
			printf(": %f%%\n", (int)(i * 0.5 * 100), j * 20, (1 - minVal) * 100);    

			if(minVal < 0.065) // 1 - 0.065 = 0.935 : 93.5% 

			{

				tempLoc.x = minLoc.x + tempTemplateWidth;
				tempLoc.y = minLoc.y + tempTemplateHeight;
				cvRectangle(destinationImage, minLoc, tempLoc, CV_RGB(0, 255, 0), 1, 8, 0);

			}

		}

		//cvShowImage(windowNameSource, result);
		//cvWaitKey(0);

		cvReleaseImage(&tempBinaryTemplateImage);
		cvReleaseImage(&result);

	}

	cvShowImage(windowNameDestination, destinationImage);
	key = cvWaitKey(0);


	cvReleaseImage(&sourceImage);
	cvReleaseImage(&templateImage);
	cvReleaseImage(&graySourceImage);
	cvReleaseImage(&grayTemplateImage);
	cvReleaseImage(&binarySourceImage);
	cvReleaseImage(&binaryTemplateImage);
	cvReleaseImage(&destinationImage);

	cvDestroyWindow(windowNameSource);
	cvDestroyWindow(windowNameDestination);
	cvDestroyWindow(windowNameCoefficientOfCorrelation);

}

void trackObject(IplImage* imgThresh){
	// Calculate the moments of 'imgThresh'
	CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
	cvMoments(imgThresh, moments, 1);
	double moment10 = cvGetSpatialMoment(moments, 1, 0);
	double moment01 = cvGetSpatialMoment(moments, 0, 1);
	double area = cvGetCentralMoment(moments, 0, 0);

	// if the area<1000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
	if(area>10 && area <3000){
		// calculate the position of the ball
		int posX = moment10/area;
		int posY = moment01/area;        

		if(lastX>=0 && lastY>=0 && posX>=0 && posY>=0)
		{
			// Draw a yellow line from the previous point to the current point
			cvLine(imgTracking, cvPoint(posX, posY), cvPoint(lastX, lastY), cvScalar(0,0,255), 4);
		}

		lastX = posX;
		lastY = posY;
	}

	free(moments); 
}

IplImage* imfill(IplImage* src)
{


	IplImage* dst = cvCreateImage(cvSize(640,480), IPL_DEPTH_8U, 1); //cvCreateImage( cvGetSize(src), 8, 3);


	CvMemStorage* storageT = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	CvSeq* largest = 0;
	float area = 0;
	int filterSize = 5;
	IplConvKernel *convKernel = cvCreateStructuringElementEx(filterSize, filterSize, (filterSize - 1) / 2, (filterSize - 1) / 2, CV_SHAPE_RECT, NULL);
	//cvErode(image,image,convKernel,2); 
	cvMorphologyEx(image, image, NULL, convKernel, CV_MOP_OPEN);
	cvReleaseStructuringElement(&convKernel); 


	cvFindContours( src, storageT, &contour, sizeof(CvContour),CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE ); //CV_RETR_EXTERNAL
	cvZero( dst );

	for( ; contour != 0; contour = contour->h_next )
	{
		float areaT = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
		if(areaT > area){
			largest = contour;
			area = areaT;
		}
	}
	if(largest!=0){
		cvDrawContours( dst, largest, cvScalar(100, 100, 100), cvScalar(100, 100, 100), 0, CV_FILLED);
		redR = cvBoundingRect(largest);
	}
	IplImage* bin_imgFilled = cvCreateImage(cvSize(640,480), IPL_DEPTH_8U, 1); //cvCreateImage(cvGetSize(src), 8, 1);
	cvInRangeS(dst, cvScalar(100, 100, 100), cvScalar(100, 100, 100), bin_imgFilled);
	if(storageT)
		cvReleaseMemStorage(&storageT);



	return bin_imgFilled;
}

//This function threshold the HSV image and create a binary image
IplImage* GetThresholdedImage(IplImage* imgHSV){     

	IplImage* imgThresh=cvCreateImage(cvGetSize(imgHSV),IPL_DEPTH_8U, 1); 
	cvInRangeS(imgHSV, cvScalar(100, 100, 100), cvScalar(110, 255, 255), imgThresh); //b g r
	imgThresh = imfill(imgThresh);

	return imgThresh;
}

Mat getEnhanced(Mat src){
	// Read source image in grayscale mode
	Mat img = src.clone();//imread("roi.png", CV_LOAD_IMAGE_GRAYSCALE);

	// Apply ??? algorithm from http://stackoverflow.com/a/14874992/2501769
	Mat enhanced, float_gray, blur, num, den;
	img.convertTo(float_gray, CV_32F, 1.0/255.0);
	cv::GaussianBlur(float_gray, blur, Size(0,0), 10);
	num = float_gray - blur;
	cv::GaussianBlur(num.mul(num), blur, Size(0,0), 20);
	cv::pow(blur, 0.5, den);
	enhanced = num / den;
	cv::normalize(enhanced, enhanced, 0.0, 255.0, NORM_MINMAX, -1);
	enhanced.convertTo(enhanced, CV_8UC1);

	// Low-pass filter
	Mat gaussian;
	cv::GaussianBlur(enhanced, gaussian, Size(0,0), 3);

	// High-pass filter on computed low-pass image
	Mat laplace;
	Laplacian(gaussian, laplace, CV_32F, 19);
	double lapmin, lapmax;
	minMaxLoc(laplace, &lapmin, &lapmax);
	double scale = 127/ max(-lapmin, lapmax);
	laplace.convertTo(laplace, CV_8U, scale, 128);

	// Thresholding using empirical value of 150 to create a vein mask
	Mat mask;
	cv::threshold(laplace, mask, THRESHOLD, 255, CV_THRESH_BINARY);

	// Clean-up the mask using open morphological operation
	morphologyEx(mask,mask,cv::MORPH_OPEN,
		getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));

	// Connect the neighboring areas using close morphological operation
	Mat connected;
	morphologyEx(mask,mask,cv::MORPH_CLOSE,
		getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11,11)));

	// Blurry the mask for a smoother enhancement
	cv::GaussianBlur(mask, mask, Size(15,15), 0);

	// Blurry a little bit the image as well to remove noise
	cv::GaussianBlur(enhanced, enhanced, Size(3,3), 0);

	// The mask is used to amplify the veins
	Mat result(enhanced);
	ushort new_pixel;
	double coeff;
	for(int i=0;i<mask.rows;i++){
		for(int j=0;j<mask.cols;j++){
			coeff = (1.0-(mask.at<uchar>(i,j)/255.0))*BRIGHT + (1-DARK);
			new_pixel = coeff * enhanced.at<uchar>(i,j);
			result.at<uchar>(i,j) = (new_pixel>255) ? 255 : new_pixel;
		}
	}

	// Show results
	imshow("result", result);


	return result;

}



Mat getEdge(Mat testImg){
	Mat grad_x,grad_y;


	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	//	double min;
	//double max;
	//cv::minMaxIdx(testImg, &min, &max);
	//cv::convertScaleAbs(testImg, testImg, 255 / max);

	/// Gradient X
	cv::Sobel( testImg, grad_x, ddepth, 1, 0, 3, scale, delta,BORDER_DEFAULT);   
	cv::convertScaleAbs( grad_x, grad_x );
	threshold(grad_x, grad_x, 20, 255, THRESH_BINARY);

	/// Gradient Y  
	cv::Sobel( testImg, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );   
	cv::convertScaleAbs( grad_y, grad_y );
	threshold(grad_y, grad_y, 20, 255, THRESH_BINARY);
	/// Total Gradient (approximate)
	//cv::addWeighted( grad_x, 0.5, grad_y, 0.5, 0, testImg );
	grad_x.copyTo(grad_y, grad_x);
	dilate(grad_y,grad_y,getStructuringElement(MORPH_OPEN,Size(3,3))); 
	//morphologyEx(grad_y,grad_y,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)));
	//cv::Canny(grad_y, grad_y, 100, 200,3,false); //100,200
	//erode(grad_y,grad_y,getStructuringElement(MORPH_OPEN,Size(3,3))); 
	//cvtColor(depthViz,binary,CV_RGB2GRAY);
	threshold(grad_y,grad_y,50,255,CV_THRESH_BINARY);

	imshow("grad_y",grad_y);
	std::vector<std::vector<Point>> contoursE;
	std::vector<Vec4i> hierarchyE;
	findContours(grad_y, contoursE, hierarchyE,
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//Mat testing;
	//testing =Scalar(0,0,0);
	drawContours(grad_y, contoursE, -1, Scalar(255, 255, 255));
	imshow("testing",grad_y);
	return grad_y;
}

void setDepthThreshold(int pos)
{
	Thresholdness = pos;
}


void onChange(int pos)
{
	if(image2) cvReleaseImage(&image2);
	if(image) cvReleaseImage(&image);

	imageM2 = testM.clone();
	image2=cvCloneImage(&(IplImage)imageM2);
	image = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 1);

	//[1]sobel

	//imageM = getEdge(testM);
	/*for(int k=0, y=0; y < imageM.rows; y++) {
	for(int x=0; x < imageM.cols; x++,k++ ){
	image->imageData[k] = imageM.data[y*imageM.cols+x]; 
	}
	}*/


	//[2] default

	cvtColor(depthFrame,imageM,CV_RGB2GRAY);
	//adaptiveThreshold(imageM,imageM,255,ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY,15,-5);
	threshold(imageM,imageM,Thresholdness,255,CV_THRESH_BINARY);
	//morphologyEx(imageM,imageM,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3))); 
	//threshold(imageM,imageM,Thresholdness,255,CV_THRESH_BINARY);
	for(int k=0, y=0; y < imageM.rows; y++) {
		for(int x=0; x < imageM.cols; x++,k++ ){
			image->imageData[k] = imageM.data[(y*imageM.cols+x)]; 
		}
	}


	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contours = 0;

	int filterSize = 5;
	IplConvKernel *convKernel = cvCreateStructuringElementEx(filterSize, filterSize, (filterSize - 1) / 2, (filterSize - 1) / 2, CV_SHAPE_RECT, NULL);
	//cvErode(image,image,convKernel,2); 
	cvMorphologyEx(image, image, NULL, convKernel, CV_MOP_OPEN);
	cvReleaseStructuringElement(&convKernel); 

	/*
	•CV_RETR_EXTERNAL gives "outer" contours, so if you have (say) one contour enclosing another (like concentric circles), only the outermost is given.
	•CV_RETR_LIST gives all the contours and doesn't even bother calculating the hierarchy -- good if you only want the contours and don't care whether one is nested inside another.
	•CV_RETR_CCOMP gives contours and organises them into outer and inner contours. Every contour is either the outline of an object, or the outline of an object inside another object (i.e. hole). The hierarchy is adjusted accordingly. This can be useful if (say) you want to find all holes.
	•CV_RETR_TREE calculates the full hierarchy of the contours. So you can say that object1 is nested 4 levels deep within object2 and object3 is also nested 4 levels deep.
	*/

	cvFindContours( image, storage, &contours, sizeof(CvContour), 
		CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE ); //CV_RETR_EXTERNAL

	if(!contours) return ; 
	int length = contours->total;	
	if(length<10) return ; 
	CvPoint* point = new CvPoint[length]; 

	CvSeqReader reader;
	CvPoint pt= cvPoint(0,0);;	
	CvSeq *contour2=contours;	

	cvStartReadSeq(contour2, &reader);
	for (int i = 0; i < length; i++)
	{
		CV_READ_SEQ_ELEM(pt, reader);
		point[i]=pt;
	}
	cvReleaseMemStorage(&storage);


	for(int i=0;i<length;i++)
	{
		int j = (i+1)%length;
		cvLine( image2, point[i],point[j],CV_RGB( 0, 0, 0 ),3,8,0 ); 
	}


	float alpha=ialpha/100.0f; // Weight of continuity energy
	float beta=ibeta/100.0f;  // Weight of curvature energy 
	float gamma=igamma/100.0f; // Weight of image energy

	CvSize size; // Size of neighborhood of every point used to search the minimumm have to be odd
	size.width=3; 
	size.height=3; 
	CvTermCriteria criteria; 
	criteria.type=CV_TERMCRIT_ITER; // terminate processing after X iteration
	criteria.max_iter=100; 
	criteria.epsilon=0.1; 
	cvSnakeImage( image, point,length,&alpha,&beta,&gamma,CV_VALUE,size,criteria,0 ); 


	for(int i=0;i<length;i++)
	{
		int j = (i+1)%length;
		cvLine( image2, point[i],point[j],CV_RGB( 0, 255, 0 ),1,8,0 ); 
		cvLine( image, point[i],point[j],CV_RGB( 255, 255, 255 ),1,8,0 ); 
	}
	cvShowImage("image",image);
	delete []point;

}






static void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nThis is a demo of Masking RGB inputs with Depth Data,\n"
		"Using OpenCV version #" << CV_VERSION << "\n"
		<< endl;

	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\t0 - contour on depth image\n"
		"\t1 - active contour\n"
		"\t2 - chamfer matching\n"
		"\t3 - blob erosion\n"
		"\t4 - contour transform\n"
		"\tl - saving current frame as a left hand template image\n"
		"\t- - Move max depth threshold in by 10 mm\n"
		"\t+ - Move max depth threshold out by 10 mm\n" << endl;
}



int main(int argc, char **argv) 
{
	help();
	int hsize = 16;
	float hranges[] = {0,180};
	const float* phranges = hranges;

	bool useDepthData = true;
	bool createDepthMask = true;
	bool showFPS = true;
	bool cleanMask = false;
	bool smoothing = true;
	bool geo = false;

	// initialize the stream capture
	CaptureStream captureStream;
	int captureStatus = captureStream.initStreamCapture();
	if (captureStatus > 0)
	{
		return 3;
	}
	int currentDepth = *captureStream.getMaxDepth();
	int currentDepthMin = *captureStream.getMinDepth();

	// initialize the background mask cleaner
	BackgroundMaskCleaner maskCleaner;
	refH.create(240,320,CV_8UC3);
	pntH.create(240,320,CV_8UC3);
	interactM.create(240,320,CV_8UC3);
	testM.create(240,320,CV_8UC3);
	refH = Scalar(0,0,0);
	pntH = Scalar(0,0,0);



	Scalar active_facet_color(0, 0, 255), delaunay_color(255,255,255);
	Rect rect(0, 0, 320, 240);
	

	
	
	Mat img(rect.size(), CV_8UC3);

	img = Scalar::all(0);
	string win = "Delaunay Demo";
	imshow(win, img);






	cvNamedWindow("active contour",0); 
	//cvCreateTrackbar("Thd", "win1", &Thresholdness, 255, onChange);
	cvCreateTrackbar("alpha", "active contour", &ialpha, 100, onChange);
	cvCreateTrackbar("beta", "active contour", &ibeta, 100, onChange);
	cvCreateTrackbar("gamma", "active contour", &igamma, 100, onChange);
	cvResizeWindow("active contour",320,300);
	//cvNamedWindow("binary",0); 
	//cvCreateTrackbar("depthTH", "binary", &Thresholdness, 255, setDepthThreshold);

	while(1)
	{

		// tell the capture stream to advance to the next available frame
		bool streamAdvanceSuccess = captureStream.advanceFrame(useDepthData, createDepthMask);
		if (!streamAdvanceSuccess)
			break;

		// create an OpenCV image for working with each frame
		Mat *rgbFrame = captureStream.getCurrentRGBFrame();
		IplImage * image = new IplImage(*rgbFrame);


		Mat depthViz = *captureStream.getDepthViz();
		depthFrame = depthViz.clone(); //.copyTo(depthFrame);

	
		interactM = depthViz.clone();
		testM = depthViz.clone();
		namedWindow("depth", CV_WINDOW_AUTOSIZE);
		imshow("depth", depthFrame);






		edgeM = depthViz.clone();
	
		cvtColor(edgeM,edgeM,CV_RGB2GRAY);
		adaptiveThreshold(edgeM,edgeM,255,ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,5,-1); //ADAPTIVE_THRESH_GAUSSIAN_C, 15, -5
		imshow("edgeM",edgeM);


		//contour of the depth image
		std::vector<std::vector<Point>> contours;
		std::vector<Vec4i> hierarchy;

		cvtColor(depthViz,binary,CV_RGB2GRAY);
		//Mat testing = binary.clone();
		//int fill = floodFill(testing,captureStream.closestPnt,Scalar(0,255,0),(cv::Rect*)0,Scalar(1,1,1),Scalar(10,10,10),8);

		threshold(binary,binary,Thresholdness,255,CV_THRESH_BINARY);
		GaussianBlur(binary,binary,Size(3,3),1,0);
		erode(binary,binary,getStructuringElement(MORPH_OPEN,Size(3,3)),Point(-1,-1),3);
		medianBlur(binary,binary,3);
		//morphologyEx(binary,binary,MORPH_ELLIPSE,getStructuringElement(MORPH_ELLIPSE,Size(3,3))); //10


		findContours(binary, contours, hierarchy,
			CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		drawContours(binary, contours, -1, Scalar(255, 255, 255));



		double min;
		double max;
		cv::minMaxIdx(depthViz, &min, &max);
		cv::Mat adjMap;
		// Histogram Equalization
		float scale2 = 255 / (max-(captureStream.closestDepth));
		depthViz.convertTo(adjMap,CV_8UC1, scale2, -(captureStream.closestDepth)*scale2); 


		applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);

		cv::imshow("colored depth map", falseColorsMap);

		Mat oriM = depthViz.clone();
		Mat erodeM;
		erode(oriM,erodeM,getStructuringElement(MORPH_CROSS,Size(5,5)),Point(-1,-1),7);
		dilate(erodeM,erodeM,getStructuringElement(MORPH_CROSS,Size(5,5)),Point(-1,-1),8);
		oriM = oriM - erodeM;
		erode(oriM,oriM,getStructuringElement(MORPH_CROSS,Size(5,5)),Point(-1,-1),1);
		imshow("show",oriM);






		if(mode==0 && contours.size()>0){
			imshow("contour",binary);
			std::vector<std::vector<cv::Point>> contours_poly(contours.size());

			std::vector<std::vector<cv::Point>> hull(contours.size());
			std::vector<std::vector<int> > hullsI(contours.size());
			std::vector< std::vector<Vec4i> > defects( contours.size() );
			vector<RotatedRect> minRect( contours.size() );


			for( int i = 0; i< contours.size(); i++ )
			{
				approxPolyDP(Mat(contours[i]),contours_poly[i],11,true); //11
				convexHull(Mat(contours_poly[i]),hull[i],false);
				convexHull( Mat(contours_poly[i]), hullsI[i], false );       
				if(hullsI[i].size() > 3 )
					convexityDefects(contours_poly[i],hullsI[i],defects[i]);
				minRect[i] = minAreaRect( Mat(contours[i]) );

			}


			Scalar color = Scalar( 255,255,0 );
			vector<vector<Point>> defect_points(contours.size());
			Mat drawing = depthViz.clone();


			drawContours( drawing, contours_poly, -1, color, 2, 8, hierarchy, 0, Point() );



			Mat imageD = depthViz.clone();



			Mat grayFrame;


			std::vector<cv::Point2f> corners;
			double quality_level = 0.5;
			double min_distance = 10;
			int eig_block_size = 3;
			int use_harris = true;

			const int MAX_CORNERS = 50;

			//drawContours( grayFrame, contours_poly, -1, color, 2, 8, hierarchy, 0, Point() );
			cv::cvtColor(depthViz, grayFrame, CV_BGR2GRAY);
			cv::goodFeaturesToTrack(grayFrame,
			corners,
			MAX_CORNERS,
			quality_level,
			min_distance,
			cv::noArray(), 
			eig_block_size,
			use_harris);


			for(int i=0;i<corners.size();i++){
			circle(grayFrame,corners[i],5,Scalar(255,255,255),-1,8,0);
			}

			imshow("corners",grayFrame);




			std::vector<cv::Point> mycontours(contours_poly[0].size());
			RotatedRect minEllipse;
			for(int j=1,cnt=0;j<contours_poly[0].size();j++){
				//float dist = sqrt(pow(hullT.at(idx).at(j).x-hullT.at(idx).at(0).x,2)+pow(hullT.at(idx).at(j).y-hullT.at(idxT).at(0).y,2));
				Point org,org2;	
				if(j==1){
					org.x = contours_poly.at(0).at(0).x;
					org.y = contours_poly.at(0).at(0).y;
					mycontours.push_back(org);
					cv::circle(drawing,cv::Point(org.x,org.y),(int)5,Scalar(255,255,0),2,8,0);
					//putText(drawing,"v"+to_string(cnt++),org,FONT_HERSHEY_SIMPLEX,0.6F,Scalar(255,255,0),1,8,false);

				}


				org.x = contours_poly.at(0).at(j).x;
				org.y = contours_poly.at(0).at(j).y;
				org2.x = contours_poly.at(0).at(j-1).x;
				org2.y = contours_poly.at(0).at(j-1).y;

				if(sqrt(pow(org.x-org2.x,2)+pow(org.y-org2.y,2)) <5){
					mycontours.push_back(Point((org.x+org2.x)/2,(org.y+org2.y)/2));
					cv::circle(drawing,Point((org.x+org2.x)/2,(org.y+org2.y)/2),(int)5,Scalar(255,255,0),-1,8,0);
					//	putText(drawing,"v"+to_string(cnt++),org,FONT_HERSHEY_SIMPLEX,0.6F,Scalar(255,255,0),1,8,false);
				}else{ 
					mycontours.push_back(org);
					cv::circle(drawing,cv::Point(org.x,org.y),(int)5,Scalar(255,255,0),-1,8,0);
					//putText(drawing,"v"+to_string(cnt++),org,FONT_HERSHEY_SIMPLEX,0.6F,Scalar(255,255,0),1,8,false);
				}

			}
			for(int j=0;j<hull[0].size();j++){
				//float dist = sqrt(pow(hullT.at(idx).at(j).x-hullT.at(idx).at(0).x,2)+pow(hullT.at(idx).at(j).y-hullT.at(idxT).at(0).y,2));
				Point org,org2;
				org.x = hull.at(0).at(j).x;
				org.y = hull.at(0).at(j).y;

				cv::circle(drawing,cv::Point(org.x,org.y),(int)5,Scalar(0,255,0),2,8,0);
				putText(drawing,"v"+to_string(j),org,FONT_HERSHEY_SIMPLEX,0.6F,Scalar(0,255,0),1,8,false);

			}

			for(int k=0;k<defects[0].size();k++)
			{           

				int ind_0=defects[0][k][0];
				int ind_1=defects[0][k][1];
				int ind_2=defects[0][k][2];
				defect_points[0].push_back(contours_poly[0][ind_2]);
				//cv::circle(drawing,contours_polyT[idx][ind_0],5,Scalar(0,255,0),-1);
				//cv::circle(drawing,contours_polyT[idx][ind_1],5,Scalar(0,255,0),-1);
				//cv::circle(drawing,contours_polyT[idx][ind_2],5,Scalar(0,255,0),-1);
				//cv::line(drawing,contours_poly[0][ind_2],contours_poly[0][ind_0],Scalar(0,255,0),1);
				//cv::line(drawing,contours_poly[0][ind_2],contours_poly[0][ind_1],Scalar(0,255,0),1);
				//cv::line(drawing,contours_poly[0][ind_0],contours_poly[0][ind_1],Scalar(255,255,0),1);
				Point org;

				org.x = contours_poly[0][ind_2].x;
				org.y = contours_poly[0][ind_2].y;
				cv::circle(drawing,cv::Point(org.x,org.y),(int)5,Scalar(0,0,255),2,8,0);
				putText(drawing,"d"+to_string(k),org,FONT_HERSHEY_SIMPLEX,0.6F,Scalar(0,0,255),1,8,false);

			}
		
			imshow("convex",drawing);







		}

		if(mode==1){
			onChange(0);
			cvShowImage("active contour",image2);
		}
		//color tracking
		/*
		IplImage *imageR=0; //colorR image
		imageR=cvCreateImage(cvSize(640,480),8,3);
		cvCopy(image,imageR);
		//create a blank image and assigned to 'imgTracking' which has the same size of original video
		imgTracking=cvCreateImage(cvGetSize(imageR),IPL_DEPTH_8U, 3);
		cvZero(imgTracking); //covert the image, 'imgTracking' to black

		//red object tracking
		cvSmooth(imageR, imageR, CV_GAUSSIAN,3,3);
		IplImage* imgHSV = cvCreateImage(cvGetSize(imageR), IPL_DEPTH_8U, 3); 
		cvCvtColor(imageR, imgHSV, CV_BGR2HSV); //Change the color format from BGR to HSV


		IplImage* imgThresh = GetThresholdedImage(imgHSV);
		//IplImage* imgThresh = GetThresholdedImage(new IplImage(backproj));
		cvSmooth(imgThresh, imgThresh, CV_GAUSSIAN,3,3); //smooth the binary image using Gaussian kernel

		//track the possition of the ball
		trackObject(imgThresh);
		cvShowImage("thresh",imgThresh);

		// Add the tracking image and the frame
		cvAdd(imageR, imgTracking, imageR);
		cvShowImage("imgTracking",imgTracking);
		*/

		if(showContour && contoursL.size()>0 && contours.size()>0){

			if(mode==2){
				Mat img = binary.clone();
				Mat cimg;
				cvtColor(img, cimg, CV_GRAY2BGR);
				Mat tpl = refH.clone();

				Canny(img, img, 5, 50, 3);
				Canny(tpl, tpl, 5, 50, 3);

				vector<vector<Point> > results;
				vector<float> costs;
				int best = chamerMatching( img, tpl, results, costs );
				if( best < 0 )
				{
					cout << "not found;\n";
					return 0;
				}

				size_t i, n = results[best].size();
				for( i = 0; i < n; i++ )
				{
					Point pt = results[best][i];
					if( pt.inside(Rect(0, 0, cimg.cols, cimg.rows)) )
						cimg.at<Vec3b>(pt) = Vec3b(0, 255, 0);
				}
				imshow("chamfer matching", cimg);
			}

			if(mode==3){
				drawContours(interactM, contoursL, -1, Scalar(0, 0, 255));
				//imshow("Interface",interactM);

				Mat blobM;
				cvtColor(depthViz,blobM,CV_RGB2GRAY);
				threshold(blobM,blobM,100,255,CV_THRESH_BINARY);
				erode(blobM,blobM,getStructuringElement(MORPH_OPEN,Size(5,5)),Point(-1,-1),5);
				std::vector<std::vector<Point>> contoursB;
				std::vector<Vec4i> hierarchyB;
				findContours(blobM, contoursB, hierarchyB,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

				if(contoursB.size()>0){
					std::vector<std::vector<Point>> contoursTmp (contoursL.size());

					drawContours(blobM, contoursB, -1, Scalar(255, 255, 255));

					vector<RotatedRect> minRectB( contoursB.size() );
					vector<RotatedRect> minEllipseB( contoursB.size() );
					int leftMostIdx = -1, leftMostX = 320;
					for(int i=0;i<contoursB.size();i++)
					{

						minRectB[i] = minAreaRect( Mat(contoursB[i]) );
						if(minRectB[i].center.x < leftMostX){
							leftMostIdx = i;
							leftMostX = minRectB[i].center.x;
						}
						if( contoursB[i].size() > 5 )
						{ 
							minEllipseB[i] = fitEllipse( Mat(contoursB[i]) ); 
						}
					}
					if(minEllipseB[leftMostIdx].size.area()>0){
						float cx = erodeL.center.x;//minEllipseB[leftMostIdx].center.x;
						float cy = erodeL.center.y;//minEllipseB[leftMostIdx].center.y;
						double angleB = minEllipseB[leftMostIdx].angle - erodeL.angle;
						angleB = angleB * atan(1)*4 / 180; 
						for(int i=0; i<contoursL.size();i++){
							for(int j=0;j < contoursL[i].size();j++){
								float s = sin(angleB);
								float c = cos(angleB);
								Point p = contoursL[i].at(j);
								// translate point back to origin:
								//	p.x -= erodeL.center.x;
								//	p.y -= erodeL.center.y;

								p.x -= cx;
								p.y -= cy;

								// rotate point
								float xnew = p.x * c - p.y * s;
								float ynew = p.x * s + p.y * c;

								// translate point back:


								p.x = xnew + cx;
								p.y = ynew + cy;

								float rescale= 1; //((float)(minEllipseB[0].size.height))/((float)(erodeL.size.height));
								p.x -= (cx-minEllipseB[leftMostIdx].center.x)*rescale;
								p.y -= (cy-minEllipseB[leftMostIdx].center.y)*rescale;

								contoursTmp[i].push_back(p);
							}
						}
					}
					Mat blobResult = depthViz.clone();
					//cvtColor(blobM, blobResult, CV_GRAY2BGR);
					ellipse( blobResult, minEllipseB[leftMostIdx],  Scalar(0,255,0), 1, 8 );
					circle(blobResult, erodeL.center, 5, Scalar(255,0,0),1,8);
					circle(blobResult, minEllipseB[leftMostIdx].center, 5, Scalar(0,255,0),1,8);
					circle(blobResult, ellipseL.center, 5, Scalar(0,0,255),1,8);
					drawContours(blobResult, contoursTmp, -1, Scalar(0, 255, 0),3,8);
					cv::imshow("blob erosion",blobResult);
				}
			}


			if(mode==4){
				std::vector<std::vector<Point>> contoursTmp (contoursL.size());
				//drawContours(interactM, contoursL, -1, Scalar(0, 0, 255));
				int idxI = -1, idxJ = -1, leftMostX = 320,bottomMostY = 230,topMostY=240,topMostX;
				for(int i=0; i<contours.size();i++){
					for(int j=0;j < contours[i].size();j++){

						if(contours[i][j].x < leftMostX && contours[i][j].y >=bottomMostY){
							idxI = i;
							idxJ = j;
							leftMostX = contours[i][j].x;
							bottomMostY = contours[i][j].y;
						}
						if(contours[i][j].y <topMostY){
							topMostY = contours[i][j].y;
							topMostX = contours[i][j].x;
						}
					}
				}
				if(idxI == -1 || idxJ == -1)
					continue;
				for(int i=0; i<contoursL.size();i++){
					for(int j=0;j < contoursL[i].size();j++){
						float cx = contoursL[leftI][leftJ].x - contours[idxI][idxJ].x;
						float cy = contoursL[leftI][leftJ].y - contours[idxI][idxJ].y;
						Point p = contoursL[i].at(j);
						//p.x -= cx;
						//p.y -= cy;
						float rescale= 1; //((float)sqrt(pow(leftMostX-topMostX,2)+pow(bottomMostY-topMostY,2)))/((float)sqrt(pow(leftX-topX,2)+pow(bottomY-topY,2)));
						p.x = (p.x -cx)*rescale;
						p.y = (p.y -cy)*rescale;


						contoursTmp[i].push_back(p);
					}
				}
				//printf("%f // %f = %f\n",((float)sqrt(pow(leftMostX-topMostX,2)+pow(bottomMostY-topMostY,2))),((float)sqrt(pow(leftX-topX,2)+pow(bottomY-topY,2))),
				//	((float)sqrt(pow(leftMostX-topMostX,2)+pow(bottomMostY-topMostY,2)))/((float)sqrt(pow(leftX-topX,2)+pow(bottomY-topY,2))));


				Mat blobTran = depthViz.clone();
				//circle(blobTran,contours[idxI][idxJ],5,Scalar(0,0,255),3,8);
				//circle(blobTran,contoursL[leftI][leftJ],5,Scalar(0,255,0),3,8);
				drawContours(blobTran, contoursTmp, -1, Scalar(0, 255, 0),3,8);
				cv::imshow("contour transform",blobTran);

			}
		}

		//http://eric-yuan.me/active-contour-snakes/



		// check for user key strokes
		char c = (char)waitKey(1);

		// if they hit the ESC key, bust out of the endless loop
		if (c == 27)
		{
			cout << "Quitting demo......." << endl;
			break;
		}
		switch(c)
		{
		case 'l':
			if(contours.size()>0){
				Mat erodeM;
				cvtColor(depthViz,erodeM,CV_RGB2GRAY);
				threshold(erodeM,erodeM,100,255,CV_THRESH_BINARY);
				erode(erodeM,erodeM,getStructuringElement(MORPH_OPEN,Size(5,5)),Point(-1,-1),5);
				std::vector<std::vector<Point>> contoursE;
				std::vector<Vec4i> hierarchyE;
				findContours(erodeM, contoursE, hierarchyE,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
				
				if(contoursE.size()>0){
					vector<RotatedRect> minEllipseE( contoursE.size() );

					for(int i=0;i<contoursE.size();i++)
					{
						
						if( contoursE[i].size() > 5 )
						{ 
							minEllipseE[i] = fitEllipse( Mat(contoursE[i]) ); 
						}
					}
					erodeL = minEllipseE[0];
				}


				contoursL.clear();
					std::vector<std::vector<cv::Point>> contours_polyL(contours.size());
				refH = Scalar(0,0,0);
				leftX = 320;
				bottomY = 230,topY=240;
				for(int i=0;i<contours.size();i++){
					contoursL.push_back(contours.at(i));
					approxPolyDP(Mat(contours[i]),contours_polyL[i],11,true); //11
					for(int j=0;j < contours[i].size();j++){
						if(leftX > contours[i][j].x && contours[i][j].y >= bottomY){
							leftI = i;
							leftJ = j;
							leftX = contours[i][j].x; 			
							bottomY  = contours[i][j].y;
						}
						if(topY > contours[i][j].y){
							topY = contours[i][j].y;
							topX = contours[i][j].x;
						}
					}

				}
				drawContours(refH, contoursL, -1, Scalar(255, 255, 255));

				std::vector<cv::Rect> boundRectL (contoursL.size());
				vector<RotatedRect> minRectL( contoursL.size() );
				vector<RotatedRect> minEllipseL( contoursL.size() );

				for(int i=0;i<contoursL.size();i++)
				{
					//boundRectL[i] = boundingRect (Mat(contoursL[i]));
					minRectL[i] = minAreaRect( Mat(contoursL[i]) );
					if( contoursL[i].size() > 5 )
					{ 
						minEllipseL[i] = fitEllipse( Mat(contoursL[i]) ); 

					}
				}
				//rectangle( refH, boundRectL[0].tl(), boundRectL[0].br(), Scalar(255,255,0), 1, 8, 0 );
				if(minEllipseL[0].size.area()>0){
					//ellipse( refH, minEllipseL[0],  Scalar(255,255,0), 1, 8 );
					ellipseL = minEllipseL[0];
				}
				// rotated rectangle
				//Point2f rect_points[4]; minRectL[0].points( rect_points );
				//for( int j = 0; j < 4; j++ )
				//	line( refH, rect_points[(j)%4], rect_points[(j+1)%4], Scalar(255,255,0), 3, 8 );
				//img = Scalar::all(0);
				
				//drawContours(img, contoursL, -1, delaunay_color);

				Subdiv2D subdiv(rect);
				paint_voronoi( img, subdiv );
				for( int i = 0; i < contours_polyL.size(); i++ )
					for(int j=0;j<contours_polyL[i].size();j++)
				{
					//Point2f fp( (float)(rand()%(rect.width-10)+5),(float)(rand()%(rect.height-10)+5));
					Point2f fp( contours_polyL[i].at(j));

					//locate_point( img, subdiv, fp, active_facet_color );
					//imshow( win, img );

					if( waitKey( 100 ) >= 0 )
						break;

					subdiv.insert(fp);
					

					//img = Scalar::all(0);
					//draw_subdiv( img, subdiv, delaunay_color );
					//imshow( win, img );

					if( waitKey( 100 ) >= 0 )
						break;
				}
				
				img = Scalar::all(0);
				draw_subdivPre( img, subdiv, delaunay_color );
				//paint_voronoi( img, subdiv );
				drawContours(img, contours_polyL, -1, Scalar(0, 255, 0));
				
				imshow( win, img );

				

				imshow("reference hand",refH);
			}
			break;
		case '0':
			mode = 1;
			break;
		case '1':
			mode = 1;
			break;
		case '2':
			mode = 2;
			break;
		case '3':
			mode = 3;
			break;
		case '4':
			mode = 4;
			break;
		case '\r':
			
			imwrite("handC"+to_string(imageNum)+".jpg",*rgbFrame);
			imwrite("handD"+to_string(imageNum)+".jpg",depthViz);
			imwrite("handE"+to_string(imageNum)+".jpg",edgeM);
					
			printf("Hand image #%d saved\n",imageNum);
			imageNum++;
			break;
		case 'g':
			geo = !geo;
			break;
		case 'f':
			// toggle the fps display
			showFPS = !showFPS;
			break;
		case '-':
			// move the max depth in
			currentDepth = *captureStream.getMaxDepth() - 10;
			captureStream.setMaxDepth(&currentDepth);
			printf("depthMax:%d\n",currentDepth);
			break;
		case '+':
			// move the max depth out
			currentDepth = *captureStream.getMaxDepth() + 10;
			captureStream.setMaxDepth(&currentDepth);
			printf("depthMax:%d\n",currentDepth);
			break;
		case '<':
			// move the max depth in
			currentDepthMin = *captureStream.getMinDepth() - 10;
			captureStream.setMinDepth(&currentDepthMin);
			printf("depthMin:%d\n",currentDepthMin);
			break;
		case '>':
			// move the max depth out
			currentDepthMin = *captureStream.getMinDepth() + 10;
			captureStream.setMinDepth(&currentDepthMin);
			printf("depthMin:%d\n",currentDepthMin);
			break;
		}

	}

	return 0;
}
