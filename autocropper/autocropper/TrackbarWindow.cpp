#include "TrackbarWindow.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>

using namespace cv;
using namespace std;
using namespace utility;

TrackbarWindow::TrackbarWindow(const string& windowName, const string& trackbarName, const int sliderDefault, const int sliderMax, Mat(*trackbarBody)(Mat image, int sliderValue))
{
	_windowName = windowName;
	_trackbarName = trackbarName;
	_sliderValue = sliderDefault;
	_sliderMax = sliderMax;
	_trackbarBody = trackbarBody;
}

void TrackbarWindow::show(const Mat& image)
{
	_image = image;
	namedWindow(_windowName, 0);

	createTrackbar(_trackbarName, _windowName, &_sliderValue, _sliderMax, &onTrackbar, (void*)this);

	onTrackbar(_sliderValue, (void*)this);
}

void TrackbarWindow::onTrackbar(int trackbarPosition, void* userData)
{
	TrackbarWindow trackbarWindow = *(TrackbarWindow*)userData;

	Mat dst = trackbarWindow._trackbarBody(trackbarWindow._image, trackbarWindow._sliderValue);

	imshow(trackbarWindow._windowName, dst);
}
