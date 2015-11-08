#pragma once

#include <opencv2/core.hpp>
#include <vector>

//////////////////////////////////////////////////////////////////////////////////
// This file contains functions that are made temporarily for experimentation on
// the images to process.
//////////////////////////////////////////////////////////////////////////////////

namespace experimental
{
	cv::Mat computeAverageImage(const std::vector<cv::Mat>& image);
	cv::Mat computeGradientImage(cv::Mat image);
	cv::Mat drawRedRectOnImage(cv::Mat image, cv::Rect rect, int thickness = 1);
	cv::Rect computeInnermostRectangle(cv::Mat image);
	cv::Rect computeGelLocation(cv::Mat image);
	cv::Mat findLargestHorizontalLines(cv::Mat image, const double percentOfWidth);
	cv::Mat findLargestVerticalLines(cv::Mat image, const double PercentOfHeight);
	cv::Mat computeForegroundImage(const std::vector<cv::Mat>& images);
	std::vector<cv::Mat> computeForegroundImages(const std::vector<cv::Mat>& images);
	cv::Mat computeHistogram(cv::Mat image);
	cv::Mat plotHistogram(cv::Mat image);

	cv::Mat generateEnhancedCenterMask(cv::Size size);
}
