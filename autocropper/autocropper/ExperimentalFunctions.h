#pragma once

#include <opencv2/core.hpp>
#include <vector>

//////////////////////////////////////////////////////////////////////////////////
// This file contains functions that are made temporarily for experimentation on
// the images to process.
//////////////////////////////////////////////////////////////////////////////////

namespace experimental
{
	cv::Mat drawRedRectOnImage(cv::Mat image, cv::Rect rect, int thickness = 1);
	cv::Rect computeInnermostRectangle(cv::Mat image);
	cv::Rect computeGelLocation(cv::Mat image);
	cv::Mat findLargestHorizontalLines(cv::Mat image);
	cv::Mat findLargestVerticalLines(cv::Mat image);
	cv::Mat computeForegroundImage(const std::vector<cv::Mat>& images);
	std::vector<cv::Mat> computeForegroundImages(const std::vector<cv::Mat>& images);
	cv::Mat computeHistogram(cv::Mat image);
	cv::Mat plotHistogram(cv::Mat image);

	cv::Mat generateEnhancedCenterMask(cv::Size size);
}
