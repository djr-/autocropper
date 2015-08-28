#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/video/background_segm.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <sys/stat.h>

using namespace std;
using namespace cv;

const int NUMBER_OF_DIGITS_IN_FILENAME = 3;	// Expected file names range from 001.png to 999.png

bool fileExists(const string& fileName)
{
	struct stat buffer;
	return stat(fileName.c_str(), &buffer) == 0;
}

vector<Mat> readDataset(const string& startingImageFilename)
{
	vector<Mat> images;

	string filename = startingImageFilename;
	string filenameSuffix = startingImageFilename.substr(startingImageFilename.find_last_of("."));
	string filenamePrefix = startingImageFilename.substr(0, startingImageFilename.find_first_of("_") + 1);
	int fileNumber = 1;

	//TODO_DR: Make this function more robust -- don't just read up to 72 and instead count how many files we will have with the specified prefix.
	while (fileNumber <= 72)
	{
		if (!fileExists(filename))
		{
			cerr << "File not found: " << filename << endl;
		}
		else
		{
			Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

			images.push_back(image);
		}

		stringstream imageNumberStream;
		fileNumber++;
		imageNumberStream << setw(NUMBER_OF_DIGITS_IN_FILENAME) << setfill('0') << fileNumber;
		filename = filenamePrefix + imageNumberStream.str() + filenameSuffix;
		imageNumberStream.str("");
	}

	cout << "Number of images read: " << images.size() << endl;

	return images;
}

int main(int argc, char** argv)
{
	if (argv[1] == 0)
	{
		cerr << "No starting file specified." << endl;
		cerr << "Exiting..." << endl;
		return EXIT_FAILURE;
	}

	if (!fileExists(argv[1]))
	{
		cerr << "Specified starting file doesn't exist: " << argv[1] << endl;
		cerr << "Exiting..." << endl;
		return EXIT_FAILURE;
	}

	vector<Mat> images = readDataset(argv[1]);
	Mat fgMask, fgImg, bgImage;
	BackgroundSubtractorMOG bg_model;

	for each (Mat image in images)
	{
		if (fgImg.empty())
			fgImg.create(image.size(), image.type());

		bg_model(image, fgMask, -10);

		fgImg = Scalar::all(0);
		image.copyTo(fgImg, fgMask);

		Mat bgImage;
		bg_model.getBackgroundImage(bgImage);

		waitKey();
	}

	imshow("fgMask", fgMask);
	imshow("fgImg", fgImg);


	waitKey();
	return EXIT_SUCCESS;
}
