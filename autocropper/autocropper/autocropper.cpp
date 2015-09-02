#include "file_utilities.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <iomanip>
#include <string>

using namespace std;
using namespace cv;
using namespace utility;

const string FILENAME_DELIMETER = "_";		// Files are expected to be named in the following format: [Prefix][Delimeter][Image Number].[File Extension]
const int NUMBER_OF_DIGITS_IN_FILENAME = 3;	// Expected file names range from 001.png to 999.png

string getFormattedFileNumber(int fileNumber);
vector<Mat> readDataset(const string& startingImageFilename);
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

	vector<Mat> images = readDataset(argv[1]);

	computeForegroundMask(images);

	return EXIT_SUCCESS;
}

string getFormattedFileNumber(int fileNumber)
{
	stringstream imageNumberStream;

	imageNumberStream << setw(NUMBER_OF_DIGITS_IN_FILENAME) << setfill('0') << fileNumber;

	return imageNumberStream.str();
}

vector<Mat> readDataset(const string& startingImageFilename)
{
	vector<Mat> images;

	string filename = startingImageFilename;
	string filenameSuffix = startingImageFilename.substr(startingImageFilename.find_last_of("."));
	string filenamePrefix = startingImageFilename.substr(0, startingImageFilename.find_first_of(FILENAME_DELIMETER) + 1);
	int fileNumber = 1;	// Assume that our starting image starts at 1.

	while (fileNumber <= 72)	//TODO_DR: Don't just read up to 72 and instead count how many files we will have with the specified prefix.
	{
		if (!FileUtilities::fileExists(filename))
		{
			cerr << "File not found: " << filename << endl;
		}
		else
		{
			Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

			images.push_back(image);
		}

		filename = filenamePrefix + getFormattedFileNumber(++fileNumber) + filenameSuffix;
	}

	cout << "Number of images read: " << images.size() << endl;

	return images;
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
