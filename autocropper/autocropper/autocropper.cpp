#include "ExperimentalFunctions.h"
#include "FileUtilities.h"
#include "ImageReader.h"
#include "OcvUtilities.h"
#include "TrackbarWindow.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
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

Rect computeVerticalContainerBoundaries(Mat originalImage)
{
	Mat verticalContainerBoundaries = findLargestVerticalLines(originalImage, 0.65);
	imwrite("TestImages/DEBUG/VerticalContainerLines.png", verticalContainerBoundaries);
	
	// If the container is not centered perfectly in front of the camera (which is likely),
	// then the vertical edges of the container will appear as thick lines in the background subtracted image.
	Rect verticalContainerRegion = computeInnermostRectangle(verticalContainerBoundaries);
	
	return verticalContainerRegion;
}

Rect computeHorizontalContainerBoundaries(Mat verticalContainerImage)
{
	Mat horizontalContainerBoundaries = findLargestHorizontalLines(verticalContainerImage, 0.9);
	imwrite("TestImages/DEBUG/HorizontalContainerLines.png", horizontalContainerBoundaries);
	Rect horizontalContainerRegion = computeInnermostRectangle(horizontalContainerBoundaries);

	return horizontalContainerRegion;
}

Rect computeContainerRegion(Mat originalImage)
{
	//Mat horiz = findLargestHorizontalLines(originalImage);
	//Mat vert = findLargestVerticalLines(originalImage);
	//Mat containerLines;
	//bitwise_or(horiz, vert, containerLines);
	//imwrite("TestImages/DEBUG/ContainerLines.png", containerLines);

	//Rect containerRegion = computeInnermostRectangle(containerLines);

	//return containerRegion;
	//bitwise_not(originalImage, originalImage);

	Rect verticalContainerLines = computeVerticalContainerBoundaries(originalImage);
	Mat verticalContainerImage = originalImage(verticalContainerLines);
	imwrite("TestImages/DEBUG/VerticalContainerImage.png", verticalContainerImage);

	auto elem = getStructuringElement(MORPH_RECT, Size(11, 9));
	Mat tmpVerticalContainerImage;
	morphologyEx(verticalContainerImage, tmpVerticalContainerImage, MORPH_CLOSE, elem);

	Rect horziontalContainerLines = computeHorizontalContainerBoundaries(tmpVerticalContainerImage);
	Mat containerImage = verticalContainerImage(horziontalContainerLines);
	imwrite("TestImages/DEBUG/ContainerImage.png", containerImage);

	//TODO: Test with new image.
	//Mat img = imread("C:/Users/Drew/Desktop/Image sets for Drew/AtTSTp0001d14_001.PNG", CV_LOAD_IMAGE_GRAYSCALE);
	//Rect newRect = Rect(verticalContainerLines.x + horziontalContainerLines.x, verticalContainerLines.y + horziontalContainerLines.y, horziontalContainerLines.width, horziontalContainerLines.height);
	////Mat tst = img(newRect);

	//Mat foregroundMask, foregroundImage, backgroundImage;
	//Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2(500, 12);

	//vector<Mat> foregroundImages;

	//int i = 0;
	//vector<Mat> images = ImageReader::readDataset("C:/Users/Drew/Desktop/Image sets for Drew/AtTSTp0001d14_001.PNG");
	//for each (Mat image in images)
	//{
	//	image = image(newRect);
	//	if (foregroundImage.empty())
	//		foregroundImage.create(image.size(), image.type());

	//	pMOG2->apply(image, foregroundMask);

	//	foregroundImage = Scalar::all(0);
	//	image.copyTo(foregroundImage, foregroundMask);	//TODO: Does using the mask rather than the image improve the results?

	//	string filename = utility::FileUtilities::buildFilename("TestImages/DEBUG/foreground2/", ++i);
	//	if (i > 1)	//TODO_DR: Deal with the first file.
	//	{
	//		imwrite(filename, foregroundImage);
	//		foregroundImages.push_back(foregroundImage.clone());
	//	}
	//}
	//Mat orImg = or(foregroundImages);
	//imwrite("testor.png", orImg);

	keepOnlyLargestContour(containerImage);
	imwrite("TestImages/DEBUG/PossibleRootSystem.png", containerImage);

	return horziontalContainerLines;	//TODO_DR: Remove.
}

Rect computeGelRegion(Mat containerImage)
{
	imwrite("TestImages/DEBUG/_ContainerFinal.png", containerImage);

	auto elem = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(containerImage, containerImage, MORPH_CLOSE, elem);
	morphologyEx(containerImage, containerImage, MORPH_OPEN, elem);
	imwrite("TestImages/DEBUG/GelAfterOpen.png", containerImage);
	
	bitwise_not(containerImage, containerImage);
	threshold(containerImage, containerImage, 245, 255, CV_THRESH_BINARY);
	OcvUtility::keepOnlyLargestContour(containerImage);

	Rect gelRegion = computeGelLocation(containerImage);

	return gelRegion;
}

Rect computeRootRegion(Mat gelImage)
{
	imwrite("TestImages/DEBUG/_GelFinal.png", gelImage);
	bitwise_not(gelImage, gelImage);
	keepOnlyLargestContour(gelImage);
	Rect rootRegion = computeGelLocation(gelImage);	//TODO: Compute root locations here, not gel location...
	imwrite("TestImages/DEBUG/_RootsFINAL.png", gelImage);
	
	return rootRegion;
}

Mat preprocessImage(Mat img)
{
	Mat orig = img.clone();
	imwrite("TestImages/1foregroundORImage.png", orig);

	Rect containerRegion = computeContainerRegion(img);
	imwrite("TestImages/2highlightedContainer.png", drawRedRectOnImage(orig, containerRegion, 3));
	Mat containerImage = img(containerRegion);

	//Rect gelRegion = computeGelRegion(img(containerRegion));
	//Rect gelRegionWRTorig = Rect(containerRegion.x + gelRegion.x, containerRegion.y + gelRegion.y, gelRegion.width, gelRegion.height);
	//imwrite("TestImages/3highlightedGel.png", drawRedRectOnImage(orig, gelRegionWRTorig, 3));
	//Mat gelImage = containerImage(gelRegion);

	//Rect rootRegion = computeRootRegion(gelImage);
	//Rect rootRegionWRTorig = Rect(containerRegion.x + gelRegion.x + rootRegion.x, containerRegion.y + gelRegion.y + rootRegion.y, rootRegion.width, rootRegion.height);
	//imwrite("TestImages/4highlightedroots.png", drawRedRectOnImage(orig, rootRegionWRTorig, 3));
	//Mat rootImage = gelImage(rootRegion);





	////TODO: Compute the widest extents in the lower 3/4 of the image and then snap the left and right edges to that point?
	//int l = rootImage.size().width;
	//int r = 0;
	//int startingHeight = static_cast<int>(rootImage.size().height * .35);
	//for (int y = startingHeight; y < rootImage.size().height; ++y)
	//{
	//	for (int x = 0; x < rootImage.size().width; ++x)
	//	{
	//		Point currentPoint = Point(x, y);
	//		//rootImage.at<uchar>(currentPoint) = 255;	//TODO: Testing where the % line is drawn.
	//		if (rootImage.at<uchar>(currentPoint) != 0)
	//		{
	//			if (currentPoint.x < l)
	//			{
	//				l = currentPoint.x;
	//			}
	//			if (currentPoint.x > r)
	//			{
	//				r = currentPoint.x;
	//			}
	//		}
	//	}
	//}

	//rootRegionWRTorig.x += l;
	//rootRegionWRTorig.width = r - l;
	//imwrite("TestImages/55555highlightedroots.png", drawRedRectOnImage(orig, rootRegionWRTorig, 3));

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
