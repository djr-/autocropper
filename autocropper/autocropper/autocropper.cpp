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

Mat preprocessImage(Mat img)
{
	Mat orig = img.clone();
	//bitwise_not(img, img);

	//blur(img, img, Size(5, 5));
	//bitwise_not(img, img);
	//threshold(img, img, 245, 255, CV_THRESH_BINARY);
	//OcvUtility::keepOnlyLargestContour(img);

	Mat horiz = findLargestHorizontalLines(img);
	Mat vert = findLargestVerticalLines(img);
	Mat both;
	bitwise_or(horiz, vert, both);

	Rect containerRegion = computeInnermostRectangle(both);

	Mat containerHighlight = drawRedRectOnImage(orig, containerRegion, 3);
	imwrite("TestImages/2highlightedContainer.png", containerHighlight);

	Mat containerImage = img(containerRegion);
	//imwrite("origimg.png", img);
	//imwrite("croppedimg.png", croppedImage);

	//Rect gelRegion = computeGelLocation(croppedImage);	// Accident that seems to help remove the bottom parts of the jar
	//Mat gelImage = croppedImage(gelRegion);
	//imwrite("gel.png", gelImage);

	//blur(containerImage, containerImage, Size(3, 3));	//TODO: temporary mechanism to speed up some work below. This decreases accuracy and should not be included in the final iteration.
	bitwise_not(containerImage, containerImage);
	threshold(containerImage, containerImage, 245, 255, CV_THRESH_BINARY);
	OcvUtility::keepOnlyLargestContour(containerImage);

	Rect gelRegion = computeGelLocation(containerImage);
	Mat gelImage = containerImage(gelRegion);
	imwrite("TestImages/gelImage.png", gelImage);

	Rect gelWRToriginal = Rect(containerRegion.x + gelRegion.x, containerRegion.y + gelRegion.y, gelRegion.width, gelRegion.height);
	Mat gelHighlight = drawRedRectOnImage(orig, gelWRToriginal, 3);
	imwrite("TestImages/3highlightedGel.png", gelHighlight);

	bitwise_not(gelImage, gelImage);
	keepOnlyLargestContour(gelImage);
	Rect rootRegion = computeGelLocation(gelImage);	//TODO: Compute root locations here, not gel location...
	Mat rootImage = gelImage(rootRegion);

	imwrite("TestImages/rootImage.png", rootImage);
	Rect rootsWRToriginal = Rect(containerRegion.x + gelRegion.x + rootRegion.x, containerRegion.y + gelRegion.y + rootRegion.y, rootRegion.width, rootRegion.height);
	Mat rootHighlight = drawRedRectOnImage(orig, rootsWRToriginal, 3);
	imwrite("TestImages/4highlightedroots.png", rootHighlight);

	Mat imgBackup = img.clone();
	Mat enhancedCenterMask = generateEnhancedCenterMask(img.size());
	img.convertTo(img, CV_32FC1);
	img = img.mul(enhancedCenterMask);
	img.convertTo(img, CV_8UC1);

	bitwise_not(img, img);
	//blur(img, img, Size(5, 5));


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

	vector<Mat> foregroundImages = experimental::computeForegroundImages(images);
	Mat orImg = or(foregroundImages);

	imwrite("TestImages/1foregroundORImage.png", orImg);

	//Mat beforeHist = plotHistogram(orImg);
	//imshow("histBeforeProcessing", beforeHist);

	orImg = preprocessImage(orImg);
	//Mat afterHist = plotHistogram(orImg);
	//imshow("histAfterProcessing", afterHist);

	const string WINDOW_NAME = "Thresholded Image";
	TrackbarWindow tbWindow = TrackbarWindow(WINDOW_NAME, "Thresh", 100, 255, trackbarMethod);
	resizeWindow(WINDOW_NAME, 1226, 1028);
	tbWindow.show(orImg);

	waitKey();

	return EXIT_SUCCESS;
}
