#pragma once

#include <opencv2/core.hpp>
#include <vector>

//////////////////////////////////////////////////////////////////////////////////
// This file contains functions that are made temporarily for experimentation on
// the images to process.
//////////////////////////////////////////////////////////////////////////////////

namespace experimental
{
	cv::Mat and(std::vector<cv::Mat>& images);
	cv::Mat computeForegroundMask(const std::vector<cv::Mat>& images);

	void padImage(const cv::Mat& sourceImage, cv::Mat& destinationImage, const int padAmount);
	void removePadding(const cv::Mat& sourceImage, cv::Mat& destinationImage, const int padAmount);

	cv::Mat generateEnhancedCenterMask(cv::Size size);
}
