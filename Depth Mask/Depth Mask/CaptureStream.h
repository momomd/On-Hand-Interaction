//=================================
// include guard
#ifndef _CAPTURE_STREAM
#define _CAPTURE_STREAM

//=================================
// forward declared dependencies


//=================================
// included dependencies

#include <vector>
#include <opencv2\opencv.hpp>
#include <util_capture.h>
#include <pxcprojection.h>
#include <pxcmetadata.h>

//=================================
// the actual class
//=================================
// the actual class
class CaptureStream
{
	public:
	CaptureStream();
	~CaptureStream();
	
	int initStreamCapture(void);
	bool advanceFrame(bool useDepthData = true, bool createDepthMask = true);
	cv::Mat * getCurrentRGBFrame(void);
	cv::Mat * getCurrentDepthMaskFrame(void);
	cv::Mat * getDepthViz(void);
	cv::Mat * getDepthInColor(void);
	bool advanceFrameNew(bool useDepthData = true, bool createDepthMask = true);
	std::vector<unsigned short> * getRawDepthData(void);
	cv::Size getRGBFrameSize(void);
	cv::Size getDepthMaskSize(void);
	cv::Size getDepthDataSize(void);
	void destroy(void);
	int * getMaxDepth(void);
	void setMaxDepth(int *value);
	int * getMinDepth(void);
	void setMinDepth(int *value);
	cv::Point2f * getRGBFOV(void);
	cv::Point2f * getDepthFOV(void);
	cv::Point CaptureStream::getC2D(int i);
	cv::Point CaptureStream::getD2C(int i);
	cv::Point getClosestPnt(void);
	int CaptureStream::getClosestDepth(void);
	// array of depth coordinates to be mapped onto color coordinates
	PXCPoint3DF32 *pos2d;
	// array of mapped color coordinates
	PXCPointF32 *posc;
	PXCPointF32 *posc1;
	PXCPointF32 *posc2;
	float *uvmap;
		int closestDepth;
		cv::Point closestPnt;

	private:
	// objects that we need to connect to Intel PerC
	PXCSmartPtr<PXCSession> session;
	pxcStatus sts;
	UtilCapture capture;
	PXCCapture::VideoStream::ProfileInfo rgbStreamInfo;
	PXCCapture::VideoStream::ProfileInfo depthStreamInfo;
	PXCSmartPtr<PXCProjection> projection;

	
	
	// special depth values for saturated and low-confidence pixels
	pxcF32 dvalues[2];
	// projection serializable identifier
	pxcUID prj_value;
	
	// storage for depth threshold (mask boundary)
	int maxDepth;
	int minDepth;

	
	// objects that we will create after we get the streams from Intel PerC
	cv::Mat rgbFrame;
	cv::Mat depthMaskFrame;
	cv::Mat depthViz;
	cv::Mat depthInColor;
	std::vector<unsigned short> depthData;
	// storage for the field of views (rgb and depth devices)
	cv::Point2f rgbFOV;
	cv::Point2f depthFOV;
};

#endif // _CAPTURE_STREAM  