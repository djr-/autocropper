#include "ImageReader.h"
#include "FileUtilities.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace autocropper;
using namespace cv;
using namespace std;
using namespace utility;

const string ImageReader::FILENAME_DELIMETER = "_";

//////////////////////////////////////////////////////////////////////////////////
// readDataset()
//
// Read the dataset beginning with the specified starting file.
//////////////////////////////////////////////////////////////////////////////////
vector<Mat> ImageReader::readDataset(const string& startingImageFilename)
{
	vector<Mat> images;

	string filename = startingImageFilename;
	string filenameSuffix = startingImageFilename.substr(startingImageFilename.find_last_of("."));
	string filenamePrefix = startingImageFilename.substr(0, startingImageFilename.find_first_of(FILENAME_DELIMETER) + 1);
	int fileNumber = 1;

	while (fileNumber <= NUMBER_OF_IMAGES_IN_SERIES)
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

//////////////////////////////////////////////////////////////////////////////////
// getFormattedFileNumber()
//
// Converts the specified file number to the format it's expected to be in
// based on a pre-defined convention.
// Expected file names range from 001.png to 999.png
//////////////////////////////////////////////////////////////////////////////////
string ImageReader::getFormattedFileNumber(const int fileNumber)
{
	stringstream imageNumberStream;

	imageNumberStream << setw(NUMBER_OF_DIGITS_IN_FILENAME) << setfill('0') << fileNumber;

	return imageNumberStream.str();
}
