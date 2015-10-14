#pragma once

#include <opencv2/imgproc/imgproc.hpp>

//////////////////////////////////////////////////////////////////////////////////
// OcvUtility
//
// Helper functions for common computations related to OpenCV.
//////////////////////////////////////////////////////////////////////////////////
namespace OcvUtility
{
	std::vector<cv::Point> keepOnlyLargestContour(cv::Mat& originalImage);
	int getLargestContourIndex(const std::vector<std::vector<cv::Point>>& contours);

	bool isPointInImage(const cv::Mat& image, const cv::Point& point);
	bool isPointWhite(const cv::Mat& image, const cv::Point& point);
	bool isPointBlack(const cv::Mat& image, const cv::Point& point);

	std::vector<cv::Point> getNeighboringPixels(const cv::Mat& image, const cv::Point& point);

	void padImage(const cv::Mat& sourceImage, cv::Mat& destinationImage, const int padAmount = 1);
	void removePadding(const cv::Mat& sourceImage, cv::Mat& destinationImage, const int padAmount = 1);

	template<typename T> static bool isPointExpectedColor(const cv::Mat& image, const cv::Point& point, const T color);
}
