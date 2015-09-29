#include "ExperimentalFunctions.h"
#include "FileUtilities.h"
#include "ImageReader.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace autocropper;
using namespace cv;
using namespace experimental;
using namespace std;
using namespace utility;

const int thresh_slider_max = 255;
int thresh_slider;
double thresh;
Mat img, dst;
const string WINDOW_NAME = "Thresholded Image";

void on_trackbar(int, void*)
{
	thresh = thresh_slider;

	threshold(img, dst, thresh, thresh_slider_max, CV_THRESH_BINARY_INV);
	auto elem = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(dst, dst, MORPH_OPEN, elem);
	morphologyEx(dst, dst, MORPH_CLOSE, elem);

	imshow(WINDOW_NAME, dst);
}

int main(int argc, char** argv)
{
	if (argv[1] == 0)
	{
		cerr << "No starting file specified." << endl;
		cerr << "Exiting..." << endl;
		return EXIT_FAILURE;
	}

	if (!FileUtilities::fileExists(argv[1]))
	{
		cerr << "Specified starting file doesn't exist: " << argv[1] << endl;
		cerr << "Exiting..." << endl;
		return EXIT_FAILURE;
	}

	//vector<Mat> images = ImageReader::readDataset(argv[1]);
	//img = images.at(0).clone();
	img = imread(argv[1]);	//TODO: Temporary mechanism to skip reading all of the iamges.

	thresh_slider = 100;

	namedWindow(WINDOW_NAME, 0);
	resizeWindow(WINDOW_NAME, 1226, 1028);

	string TrackbarName = "Thresh";
	createTrackbar(TrackbarName, WINDOW_NAME, &thresh_slider, thresh_slider_max, on_trackbar);

	on_trackbar(thresh_slider, 0);

	waitKey();

	return EXIT_SUCCESS;
}
