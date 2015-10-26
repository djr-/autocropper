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

//Mat g_img;

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
	imwrite("TestImages/DEBUG/ContainerLines.png", containerLines);

	Rect containerRegion = computeInnermostRectangle(containerLines);

	return containerRegion;
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

	//threshold(containerImage, containerImage, 1, 255, THRESH_OTSU);
	//imwrite("otsu_container.png", containerImage);
	//Mat otsu_container = g_img(containerRegion).clone();
	//imwrite("1container.png", otsu_container);
	//Mat containerGrad = computeGradientImage(otsu_container).clone();
	//imwrite("1containerGrad.png", containerGrad);
	//imwrite("1grad_hist.png", plotHistogram(containerGrad));
	//imwrite("1container_hist.png", plotHistogram(otsu_container));
	//threshold(otsu_container, otsu_container, 1, 255, THRESH_OTSU);
	//threshold(containerGrad, containerGrad, 1, 255, THRESH_OTSU);
	//imwrite("1otsu_container.png", otsu_container);
	//imwrite("1otsu_containerGrad.png", containerGrad);

	Rect gelRegion = computeGelRegion(img(containerRegion));
	Rect gelRegionWRTorig = Rect(containerRegion.x + gelRegion.x, containerRegion.y + gelRegion.y, gelRegion.width, gelRegion.height);
	imwrite("TestImages/3highlightedGel.png", drawRedRectOnImage(orig, gelRegionWRTorig, 3));
	Mat gelImage = containerImage(gelRegion);

	//threshold(gelImage, gelImage, 1, 255, THRESH_OTSU);
	//imwrite("otsu_gel.png", gelImage);
	//Mat otsu_gel = g_img(gelRegionWRTorig).clone();
	//imwrite("2gel.png", otsu_gel);
	//Mat gelGrad = computeGradientImage(otsu_gel).clone();
	//imwrite("2gelGrad.png", gelGrad);
	//imwrite("2grad_hist.png", plotHistogram(gelGrad));
	//imwrite("2gel_hist.png", experimental::plotHistogram(otsu_gel));
	//threshold(otsu_gel, otsu_gel, 1, 255, THRESH_OTSU);
	//threshold(gelGrad, gelGrad, 1, 255, THRESH_OTSU);
	//imwrite("2otsu_gel.png", otsu_gel);
	//imwrite("2otsu_gelGrad.png", gelGrad);

	Rect rootRegion = computeRootRegion(gelImage);
	Rect rootRegionWRTorig = Rect(containerRegion.x + gelRegion.x + rootRegion.x, containerRegion.y + gelRegion.y + rootRegion.y, rootRegion.width, rootRegion.height);
	imwrite("TestImages/4highlightedroots.png", drawRedRectOnImage(orig, rootRegionWRTorig, 3));
	Mat rootImage = gelImage(rootRegion);

	//threshold(rootImage, rootImage, 1, 255, THRESH_OTSU);
	//imwrite("otsu_root.png", rootImage);
	//Mat otsu_root = g_img(rootRegionWRTorig).clone();
	//imwrite("3root.png", otsu_root);
	//Mat rootGrad = computeGradientImage(otsu_root).clone();
	//imwrite("3rootGrad.png", rootGrad);
	//imwrite("3grad_hist.png", plotHistogram(rootGrad));
	//imwrite("3root_hist.png", experimental::plotHistogram(otsu_root));
	//threshold(otsu_root, otsu_root, 1, 255, THRESH_OTSU);
	//threshold(rootGrad, rootGrad, 1, 255, THRESH_OTSU);
	//imwrite("3otsu_root.png", otsu_root);
	//imwrite("3otsu_rootGrad.png", rootGrad);

	//TODO: Compute the widest extents in the lower 3/4 of the image and then snap the left and right edges to that point?

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
	//g_img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);	//TODO: Temporary mechanism to skip reading all of the images.

	//Mat gradientImage = computeGradientImage(g_img).clone();
	//threshold(gradientImage, gradientImage, 1, 255, THRESH_OTSU);
	//imwrite("ORIG_otsu.png", gradientImage);

	vector<Mat> foregroundImages = computeForegroundImages(images);
	Mat orImg = or(foregroundImages);

	//Mat gradientImage = computeGradientImage(orImg).clone();
	//imwrite("or_grad.png", gradientImage);
	//threshold(gradientImage, gradientImage, 1, 255, THRESH_OTSU);
	//imwrite("or_otsu.png", gradientImage);

	orImg = preprocessImage(orImg);

	//const string WINDOW_NAME = "Thresholded Image";
	//TrackbarWindow tbWindow = TrackbarWindow(WINDOW_NAME, "Thresh", 100, 255, trackbarMethod);
	//resizeWindow(WINDOW_NAME, 1226, 1028);
	//tbWindow.show(orImg);
	//waitKey();

	return EXIT_SUCCESS;
}
