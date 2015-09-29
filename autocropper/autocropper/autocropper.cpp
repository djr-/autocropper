#include "file_utilities.h"
#include "image_reader.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <string>

using namespace autocropper;
using namespace std;
using namespace cv;
using namespace utility;

void computeForegroundMask(const vector<Mat>& images);

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

	vector<Mat> images = autocropper::ImageReader::readDataset(argv[1]);

	computeForegroundMask(images);

	return EXIT_SUCCESS;
}

void computeForegroundMask(const vector<Mat>& images)
{
	Mat foregroundMask, foregroundImage, backgroundImage;
	Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2();

	vector<Mat> foregroundMasks;
	
	int i = 0;

	for each (Mat image in images)
	{
		if (foregroundImage.empty())
			foregroundImage.create(image.size(), image.type());

		pMOG2->apply(image, foregroundMask);

		foregroundImage = Scalar::all(0);
		image.copyTo(foregroundImage, foregroundMask);

		string filename = FileUtilities::buildFilename("C:\\Temp\\images\\fg", ++i);
		imwrite(filename, foregroundImage);
	}

	imwrite("TestImages/tmp/fg.png", foregroundImage);

	//imshow("fgMask", foregroundMask);
	//imshow("fgImg", foregroundImage);

	waitKey();
}
