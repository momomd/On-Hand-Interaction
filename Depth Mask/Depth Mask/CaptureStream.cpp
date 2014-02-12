#include "CaptureStream.h"

using namespace cv;
using namespace std;

CaptureStream::CaptureStream() : sts(PXCSession_Create(&session)), capture(session)
{
	rgbStreamInfo.imageInfo.width = NULL;
	depthStreamInfo.imageInfo.width = NULL;
	maxDepth = 300;
	closestDepth = maxDepth;
	// rgb FOVs (H and V) in radians (default is for the Creative Gesture Cam)
	rgbFOV.x = 1.003888951f; // 57.5186 degrees
	rgbFOV.y = 0.76542287f; // 43.8555 degrees
	// depth FOVs (H and V) in radians (default is for the Creative Gesture Cam)
	depthFOV.x = 1.238419315f; // 70.9562
	depthFOV.y = 0.960044535f; // 55.0065
}

int * CaptureStream::getMaxDepth(void)
{
	return &maxDepth;
}


void CaptureStream::setMaxDepth(int *value)
{
	maxDepth = *value;
}

int * CaptureStream::getMinDepth(void)
{
	return &maxDepth;
}


void CaptureStream::setMinDepth(int *value)
{
	maxDepth = *value;
}


CaptureStream::~CaptureStream()
{
	destroy();
}

void CaptureStream::destroy(void)
{
	// delete the arrays (if they were built)
	if (pos2d) 
		delete [] pos2d;
	if (posc) 
		delete [] posc;
}

Point CaptureStream::getC2D(int i)
{
	return Point((int)pos2d[i].x,(int)pos2d[i].y);
}

Point CaptureStream::getD2C(int i)
{
	return Point((int)posc[i].x,(int)posc[i].y);
}

Point CaptureStream::getClosestPnt(void)
{
	return closestPnt;
}

int CaptureStream::getClosestDepth(void)
{
	return closestDepth;
}


int CaptureStream::initStreamCapture()
{
	// setup capture to get the desired streams
	PXCCapture::VideoStream::DataDesc request; 
	memset(&request, 0, sizeof(request)); 
	request.streams[0].format = PXCImage::COLOR_FORMAT_RGB32;
	request.streams[1].format = PXCImage::COLOR_FORMAT_DEPTH;
	pxcStatus sts = capture.LocateStreams(&request);

	if (sts < PXC_STATUS_NO_ERROR) 
	{
		cout << "Failed to locate video stream(s)" << endl;
		return 1;
	}

	// stream profile for the color stream
	capture.QueryVideoStream(0)->QueryProfile(&rgbStreamInfo);
	//cout << "rgbStreamInfo.imageInfo.width: " << rgbStreamInfo.imageInfo.width << endl;

	// stream profile for the depth stream
	capture.QueryVideoStream(1)->QueryProfile(&depthStreamInfo);
	//cout << "depthStreamInfo.imageInfo.width: " << depthStreamInfo.imageInfo.width << endl;

	// size the vector that will hold mapped depth data
	depthData.resize(rgbStreamInfo.imageInfo.width * rgbStreamInfo.imageInfo.height, (pxcU16)0);

	// allocate frames for the rgb and depth mask
	rgbFrame.create(rgbStreamInfo.imageInfo.height, rgbStreamInfo.imageInfo.width, CV_8UC3);
	// mask is one channel image
	depthMaskFrame.create(rgbStreamInfo.imageInfo.height, rgbStreamInfo.imageInfo.width, CV_8UC1);
	depthViz.create(depthStreamInfo.imageInfo.height, depthStreamInfo.imageInfo.width, CV_8UC3);

	// set the desired value for smoothing the depth data
	capture.QueryDevice()->SetProperty(PXCCapture::Device::PROPERTY_DEPTH_SMOOTHING, 1);

	// setup the projection info for getting a nice depth map
	sts = capture.QueryDevice()->QueryPropertyAsUID(PXCCapture::Device::PROPERTY_PROJECTION_SERIALIZABLE, &prj_value);
	if (sts >= PXC_STATUS_NO_ERROR) 
	{
		// create properties for checking if depth values are bad (by low confidence and by saturation)
		capture.QueryDevice()->QueryProperty(PXCCapture::Device::PROPERTY_DEPTH_LOW_CONFIDENCE_VALUE, &dvalues[0]);
		capture.QueryDevice()->QueryProperty(PXCCapture::Device::PROPERTY_DEPTH_SATURATION_VALUE, &dvalues[1]);

		session->DynamicCast<PXCMetadata>()->CreateSerializable<PXCProjection>(prj_value, &projection);

		int npoints = rgbStreamInfo.imageInfo.width * rgbStreamInfo.imageInfo.height; 
		pos2d = (PXCPoint3DF32*)new PXCPoint3DF32[npoints];
		posc = (PXCPointF32*)new PXCPointF32[npoints];
		for (int y = 0, k = 0; (pxcU32)y < depthStreamInfo.imageInfo.height; y++)
		{
			for (int x = 0; (pxcU32)x < depthStreamInfo.imageInfo.width; x++, k++)
			{
				// prepopulate the x and y values of the the depth data
				pos2d[k].x = (pxcF32)x;
				pos2d[k].y = (pxcF32)y;

			}
		}
	}

	// query for the capture device to find camera properties
	PXCCapture::Device *device = capture.QueryDevice();
	if (device)
	{
		// find and save the RGB horizontal and vertical FOV in radians
		PXCPointF32 tempRgbFov;
		sts = device->QueryPropertyAsPoint(device->PROPERTY_COLOR_FIELD_OF_VIEW, &tempRgbFov); // &rgbFOV);
		if (sts == PXC_STATUS_NO_ERROR)
		{
			rgbFOV.x = tempRgbFov.x * CV_PI / (pxcF32)180.0F;
			rgbFOV.y = tempRgbFov.y * CV_PI / (pxcF32)180.0F;
			//cout << "RGB H-FOV: " << rgbFOV.x << endl;
			//cout << "RGB V-FOV: " << rgbFOV.y << endl;
		}

		// find and save the depth horiztonal and vertical field of view in radians
		PXCPointF32 tempDepthFov;
		sts = device->QueryPropertyAsPoint(device->PROPERTY_DEPTH_FIELD_OF_VIEW, &tempDepthFov);
		if (sts == PXC_STATUS_NO_ERROR)
		{
			depthFOV.x = tempDepthFov.x * CV_PI / (pxcF32)180.0F;
			depthFOV.y = tempDepthFov.y * CV_PI / (pxcF32)180.0F;
			//cout << "DEPTH H-FOV: " << depthFOV.x << endl;
			//cout << "DEPTH V-FOV: " << depthFOV.y << endl;
		}
	}

	// everything is good, return a 0.
	return 0;
}

cv::Size CaptureStream::getRGBFrameSize(void)
{
	if (rgbStreamInfo.imageInfo.width)
		return Size(rgbStreamInfo.imageInfo.width, rgbStreamInfo.imageInfo.height);
	return Size(0,0);
}


cv::Size CaptureStream::getDepthMaskSize(void)
{
	if (rgbStreamInfo.imageInfo.width)
		return Size(rgbStreamInfo.imageInfo.width, rgbStreamInfo.imageInfo.height);
	return Size(0,0);
}


cv::Size CaptureStream::getDepthDataSize(void)
{
	if (depthStreamInfo.imageInfo.width)
		return Size(depthStreamInfo.imageInfo.width, depthStreamInfo.imageInfo.height);
	return Size(0,0);
}

/**
* Advances the stream to the next available frame
* @param bool useDepthData - indicates if the depth data will be used on this frame. If true, the data will be saved 
* in the depthData vector for other classes to use as needed
* @param bool createDepthMask - indicates if a depth mask should be created for this frame. Depth mask data will be saved as 
* an OpenCV Mat object to be used by other classes as needed
*/
bool CaptureStream::advanceFrame(bool useDepthData, bool createDepthMask)
{
	// vector of images for temp storage of:
	// [0] raw rgb frame
	// [1] raw depth frame
	PXCSmartArray<PXCImage> images(2);

	// Intel PerC utility for managing the application object instances
	// used below to help sync up the rgb and depth frames in the stream (same moment in time)
	PXCSmartSP sp;

	// setup syncing so depth and color will come in at the same instance in time
	pxcStatus sts = capture.ReadStreamAsync(images, &sp);
	if (sts < PXC_STATUS_NO_ERROR) 
		return false;

	sts = sp->Synchronize();
	if (sts < PXC_STATUS_NO_ERROR) 
		return false;

	// grab the rgb image
	PXCImage::ImageData rgbImage;
	images[0]->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::COLOR_FORMAT_RGB32, &rgbImage);
	// find rgbImage stride
	int rgbStride = rgbImage.pitches[0] / sizeof(pxcU32);






	// keep depth image stride in this scope
	int depthStride = 0;


	// Begin with all white pixels in the depth mask
	if (useDepthData && createDepthMask)
		depthMaskFrame = Scalar(255);

	// if the depth data is on
	PXCImage::ImageData depthImage;
	if (useDepthData)
	{
		// grab the depth image
		images[1]->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::COLOR_FORMAT_DEPTH, &depthImage);

		// find depth image stride
		depthStride = depthImage.pitches[0] / sizeof(pxcU16);
		uvmap=(float*)depthImage.planes[2]; 
		// setup the depth data so we can map it to color data using the projection
		for (int y = 0, k = 0; (pxcU32)y < depthStreamInfo.imageInfo.height; y++)
		{
			for (int x = 0; (pxcU32)x < depthStreamInfo.imageInfo.width; x++, k++)
			{
				// raw depth data
				pos2d[k].z = ((short*)depthImage.planes[0])[y * depthStride + x];
				if (pos2d[k].z < maxDepth)
				{
					depthViz.data[3*y*depthStride+3*x+0] = pos2d[k].z;
					depthViz.data[3*y*depthStride+3*x+1] = pos2d[k].z;
					depthViz.data[3*y*depthStride+3*x+2] = pos2d[k].z;	

				}else{
					depthViz.data[3*y*depthStride+3*x+0] = 0;//maxDepth;
					depthViz.data[3*y*depthStride+3*x+1] = 0;//maxDepth;
					depthViz.data[3*y*depthStride+3*x+2] = 0;	
				}

			}
		}

		// use the projection to map depth to color with this frame
		projection->MapDepthToColorCoordinates(rgbStreamInfo.imageInfo.width * rgbStreamInfo.imageInfo.height, pos2d, posc);
		//projection->MapColorCoordinatesToDepth(rgbStreamInfo.imageInfo.width * rgbStreamInfo.imageInfo.height,posc,pos2d);


		// save the aligned depth data so we can use it in other places
		// and create the depth mask (if the flag is on)

		for (int y = 0, k = 0; y < (int)depthStreamInfo.imageInfo.height; y++) //240
		{
			for (int x = 0; x < (int)depthStreamInfo.imageInfo.width; x++, k++) //360
			{
				int xx = (int)(posc[k].x + 0.5f);
				int yy = (int)(posc[k].y + 0.5f);
				int currentIndex = yy * depthStreamInfo.imageInfo.width + xx;
				depthData.at(currentIndex) = 0;
				if (xx < 0 || yy < 0 || (pxcU32)xx >= rgbStreamInfo.imageInfo.width || (pxcU32)yy >= rgbStreamInfo.imageInfo.height) 
					continue; // no mapping based on clipping due to differences in FOV between the two cameras.
				if (pos2d[k].z == dvalues[0] || pos2d[k].z == dvalues[1]) 
					continue; // no mapping based on unreliable depth values

				// save the mapped depth data
				depthData.at(currentIndex) = pos2d[k].z;
				// create the depth mask frame
				if (createDepthMask)
				{
					if (pos2d[k].z < maxDepth)
					{
						depthMaskFrame.data[depthMaskFrame.step[0] * yy + depthMaskFrame.step[1] * xx] = 0;
						int dist = pos2d[k].z; 
						if(dist < closestDepth && dist >0){
							closestDepth = dist;
							closestPnt = Point(x,y);

						}
					}
				}


			}
		}


		//populate gaps in picture UVMAP horizontal and verticle
		int storeX = 0, storeY=0, storeZ=0, storeCnt = 5; 
		for (int y=0;y<480;y++){
			for(int x=0;x<640;x++){
				int depthX = pos2d[y*rgbStreamInfo.imageInfo.width+x].x;
				int depthY = pos2d[y*rgbStreamInfo.imageInfo.width+x].y;
				int depthZ = pos2d[y*rgbStreamInfo.imageInfo.width+x].z;
				/*if(depthX >0 && depthY>0 && depthZ != dvalues[0] && depthZ != dvalues[1]){
					storeX = depthX;
					storeY = depthY;
					storeZ = depthZ;
					storeCnt = 3;
				}else{
					if(storeCnt > 0){
						pos2d[y*rgbStreamInfo.imageInfo.width+x].x = storeX;
						pos2d[y*rgbStreamInfo.imageInfo.width+x].y = storeY;
						pos2d[y*rgbStreamInfo.imageInfo.width+x].z = storeZ;
						storeCnt--;
					}
				}*/
				if((depthZ < maxDepth && depthZ > 0) || depthZ == dvalues[0] || depthZ == dvalues[1]){
					if(depthX >0 && depthY>0 && depthZ != dvalues[0] && depthZ != dvalues[1]){
						storeX = depthX;
						storeY = depthY;
						storeZ = depthZ;
						storeCnt = 3;
					}else{
						if(storeCnt > 0){
							pos2d[y*rgbStreamInfo.imageInfo.width+x].x = storeX;
							pos2d[y*rgbStreamInfo.imageInfo.width+x].y = storeY;
							pos2d[y*rgbStreamInfo.imageInfo.width+x].z = storeZ;
							storeCnt--;
						}
					}
				}
			}
		}
		for(int x=0;x<640;x++){
			for (int y=0;y<480;y++){
				int depthX = pos2d[y*rgbStreamInfo.imageInfo.width+x].x;
				int depthY = pos2d[y*rgbStreamInfo.imageInfo.width+x].y;
				int depthZ = pos2d[y*rgbStreamInfo.imageInfo.width+x].z;
				if((depthZ < maxDepth && depthZ > 0) || depthZ == dvalues[0] || depthZ == dvalues[1]){
					if(depthX >0 && depthY>0 && depthZ != dvalues[0] && depthZ != dvalues[1]){
						storeX = depthX;
						storeY = depthY;
						storeZ = depthZ;
						storeCnt = 3;
					}else{
						if(storeCnt > 0){
							pos2d[y*rgbStreamInfo.imageInfo.width+x].x = storeX;
							pos2d[y*rgbStreamInfo.imageInfo.width+x].y = storeY;
							pos2d[y*rgbStreamInfo.imageInfo.width+x].z = storeZ;
							storeCnt--;
						}
					}
				}
			}
		}

	}

	// create the rgb image (in opencv format)
	for (int y = 0; y < rgbFrame.rows; y++) 
	{
		//uchar* depthMaskPtr = (uchar*)(depthMask->imageData + y * depthMask->widthStep); 
		for (int x = 0; x < rgbFrame.cols; x++) 
		{
			// write the captured frame into an image that openCV can use
			int cIndex = y * rgbStride + x;
			int rgbFrameIndex = rgbFrame.step[0] * y + rgbFrame.step[1] * x;
			// set the red channel (BGR format)
			rgbFrame.data[rgbFrameIndex + 2] = ((pxcU32*)rgbImage.planes[0])[cIndex] >> 16;
			// set the green channel (BGR format)
			rgbFrame.data[rgbFrameIndex + 1] = ((pxcU32*)rgbImage.planes[0])[cIndex] >> 8 & 0xFF;
			// set the blue channel (BGR format)
			rgbFrame.data[rgbFrameIndex] = ((pxcU32*)rgbImage.planes[0])[cIndex];
		}
	}

	// release the temp rgb image 
	images[0]->ReleaseAccess(&rgbImage);

	// release the temp depth image (if we used it)
	if (useDepthData)
		images[1]->ReleaseAccess(&depthImage);

	// return true indicating everything worked
	return true;
}


Mat * CaptureStream::getCurrentRGBFrame(void)
{
	return &rgbFrame;
}

Mat * CaptureStream::getCurrentDepthMaskFrame(void)
{
	return &depthMaskFrame;
}

Mat * CaptureStream::getDepthViz(void)
{
	return &depthViz;
}

Mat * CaptureStream::getDepthInColor(void)
{
	return &depthInColor;
}



vector<unsigned short> * CaptureStream::getRawDepthData(void)
{
	return &depthData;
}

Point2f * CaptureStream::getRGBFOV(void)
{
	return &rgbFOV;
}

Point2f * CaptureStream::getDepthFOV(void)
{
	return &depthFOV;
}