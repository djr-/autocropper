#include "ExperimentalFunctions.h"
#include "FileUtilities.h"
#include "ImageReader.h"
#include "OcvUtilities.h"
#include "TrackbarWindow.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace autocropper;
using namespace cv;
using namespace experimental;
using namespace std;
using namespace utility;
using namespace OcvUtility;

Mat trackbarMethod(Mat image, int sliderValue)
{
	Mat dst;

	double thresh = sliderValue;
	threshold(image, dst, thresh, 255, CV_THRESH_BINARY_INV);
	auto elem = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(dst, dst, MORPH_OPEN, elem);
	morphologyEx(dst, dst, MORPH_CLOSE, elem);

	return dst;
}

Rect computeContainerRegion(Mat originalImage)
{
	Mat horiz = findLargestHorizontalLines(originalImage);
	Mat vert = findLargestVerticalLines(originalImage);
	Mat containerLines;
	bitwise_or(horiz, vert, containerLines);

	Rect containerRegion = computeInnermostRectangle(containerLines);

	return containerRegion;
}

Rect computeGelRegion(Mat containerImage)
{
	//blur(containerImage, containerImage, Size(3, 3));	//TODO: temporary mechanism to speed up some work below. This decreases accuracy and should not be included in the final iteration.
	bitwise_not(containerImage, containerImage);
	threshold(containerImage, containerImage, 245, 255, CV_THRESH_BINARY);
	OcvUtility::keepOnlyLargestContour(containerImage);

	Rect gelRegion = computeGelLocation(containerImage);

	return gelRegion;
}

Rect computeRootRegion(Mat gelImage)
{
	bitwise_not(gelImage, gelImage);
	keepOnlyLargestContour(gelImage);
	Rect rootRegion = computeGelLocation(gelImage);	//TODO: Compute root locations here, not gel location...
	
	return rootRegion;
}

Mat preprocessImage(Mat img)
{
	Mat orig = img.clone();
	imwrite("TestImages/1foregroundORImage.png", orig);

	Rect containerRegion = computeContainerRegion(img);
	imwrite("TestImages/2highlightedContainer.png", drawRedRectOnImage(orig, containerRegion, 3));
	Mat containerImage = img(containerRegion);

	Rect gelRegion = computeGelRegion(img(containerRegion));
	Rect gelRegionWRTorig = Rect(containerRegion.x + gelRegion.x, containerRegion.y + gelRegion.y, gelRegion.width, gelRegion.height);
	imwrite("TestImages/3highlightedGel.png", drawRedRectOnImage(orig, gelRegionWRTorig, 3));
	Mat gelImage = containerImage(gelRegion);

	Rect rootRegion = computeRootRegion(gelImage);
	Rect rootRegionWRTorig = Rect(containerRegion.x + gelRegion.x + rootRegion.x, containerRegion.y + gelRegion.y + rootRegion.y, rootRegion.width, rootRegion.height);
	imwrite("TestImages/4highlightedroots.png", drawRedRectOnImage(orig, rootRegionWRTorig, 3));
	Mat rootImage = gelImage(rootRegion);

	return img;
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

	vector<Mat> images = ImageReader::readDataset(argv[1]);
	//Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);	//TODO: Temporary mechanism to skip reading all of the images.

	vector<Mat> foregroundImages = computeForegroundImages(images);
	Mat orImg = or(foregroundImages);

	orImg = preprocessImage(orImg);

	//const string WINDOW_NAME = "Thresholded Image";
	//TrackbarWindow tbWindow = TrackbarWindow(WINDOW_NAME, "Thresh", 100, 255, trackbarMethod);
	//resizeWindow(WINDOW_NAME, 1226, 1028);
	//tbWindow.show(orImg);
	//waitKey();

	return EXIT_SUCCESS;
}
