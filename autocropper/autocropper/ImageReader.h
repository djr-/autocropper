#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace autocropper
{
	//////////////////////////////////////////////////////////////////////////////////
	// ImageReader
	//
	// Encapsulate logic related to reading and translating images into objects
	// that can be operated on via OpenCV.
	//////////////////////////////////////////////////////////////////////////////////
	class ImageReader final
	{
	public:
		static std::vector<cv::Mat> readDataset(const std::string& startingImageFilename);
	private:
		static std::string getFormattedFileNumber(const int fileNumber);

		static const std::string FILENAME_DELIMETER;		// Files are expected to be named in the following format: [Prefix][Delimeter][Image Number].[File Extension]
		static const int NUMBER_OF_DIGITS_IN_FILENAME = 3;	// Expected file names range from 001.png to 999.png
		static const int NUMBER_OF_IMAGES_IN_SERIES = 72;	// There should always be 72 in the input image series.
	};
}
