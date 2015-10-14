#include "OcvUtilities.h"

using namespace std;
using namespace cv;

namespace OcvUtility
{
	//////////////////////////////////////////////////////////////////////////////////
	// keepOnlyLargestContour()
	//
	// Remove all contours that are not the largest contour from the specified image.
	// Returns the contour that was found.
	//////////////////////////////////////////////////////////////////////////////////
	vector<Point> keepOnlyLargestContour(Mat& originalImage)
	{
		Mat largestContourImage;
		padImage(originalImage, largestContourImage);	// If we don't pad, then findContours will not mark the edge as part of the contour.

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(largestContourImage, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

		int largestContourIndex = getLargestContourIndex(contours);

		originalImage = originalImage.zeros(originalImage.size(), CV_8UC1);	// Clear the existing image before drawing the largest contour back onto it.
		largestContourImage = largestContourImage.zeros(largestContourImage.size(), CV_8UC1);

		drawContours(largestContourImage, contours, largestContourIndex, Scalar(255), CV_FILLED, 8, hierarchy);	//TODO: Instead of 255, use maximumThresholdValue

		removePadding(largestContourImage, originalImage);

		return contours[largestContourIndex];
	}

	//////////////////////////////////////////////////////////////////////////////////
	// getLargestContourIndex()
	//
	// Find the largest contour in the list of contours and return its index.
	//////////////////////////////////////////////////////////////////////////////////
	int getLargestContourIndex(const vector<vector<Point>>& contours)
	{
		int largestContourIndex = 0;
		double largestContourArea = 0;
		for (int i = 0; i < contours.size(); i++)
		{
			double currentContourArea = contourArea(contours[i]);
			if (currentContourArea > largestContourArea)
			{
				largestContourArea = currentContourArea;
				largestContourIndex = i;
			}
		}

		return largestContourIndex;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// isPointInImage()
	//
	// Returns true if the specified point is inside the image, false otherwise.
	//////////////////////////////////////////////////////////////////////////////////
	bool isPointInImage(const Mat& image, const Point& point)
	{
		return point.inside(Rect(0, 0, image.size().width, image.size().height));
	}

	//////////////////////////////////////////////////////////////////////////////////
	// isPointExpectedColor()
	//
	// Returns true if the specified point is both in the image and is the specified
	// color.
	//////////////////////////////////////////////////////////////////////////////////
	template<typename T>
	bool isPointExpectedColor(const Mat& image, const Point& point, const T color)
	{
		if (!isPointInImage(image, point))
		{
			assert(false);
			return false;
		}

		return image.at<T>(point) == color;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// isPointWhite()
	//
	// Returns true if the specified point is both in the image and is white.
	// Note that this function assumes we are representing our pixels as a uchar.
	//////////////////////////////////////////////////////////////////////////////////
	bool isPointWhite(const Mat& image, const Point& point)
	{
		return isPointExpectedColor<uchar>(image, point, 255);
	}

	//////////////////////////////////////////////////////////////////////////////////
	// isPointBlack()
	//
	// Returns true if the specified point is both in the image and is black.
	// Note that this function assumes we are representing our pixels as a uchar.
	//////////////////////////////////////////////////////////////////////////////////
	bool isPointBlack(const Mat& image, const Point& point)
	{
		return isPointExpectedColor<uchar>(image, point, 0);
	}

	//////////////////////////////////////////////////////////////////////////////////
	// getNeighboringPixels()
	//
	// Returns a list of the 8 neighboring pixels that are in the image.
	//////////////////////////////////////////////////////////////////////////////////
	vector<Point> getNeighboringPixels(const Mat& image, const Point& point)
	{
		vector<Point> neighboringPixels;

		for (int x = point.x - 1; x <= point.x + 1; ++x)
		{
			for (int y = point.y - 1; y <= point.y + 1; ++y)
			{
				Point neighbor(x, y);

				if (neighbor == point)
					continue;	// Don't add yourself as a neighbor.

				if (isPointInImage(image, neighbor))
					neighboringPixels.push_back(neighbor);
			}
		}

		return neighboringPixels;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// padImage()
	//
	// Pads the image by the specified padding amount with the default value (0).
	//////////////////////////////////////////////////////////////////////////////////
	void padImage(const Mat& sourceImage, Mat& destinationImage, const int padAmount)
	{
		copyMakeBorder(sourceImage, destinationImage, 1, 1, 1, 1, BORDER_CONSTANT);
	}

	//////////////////////////////////////////////////////////////////////////////////
	// removePadding()
	//
	// Removes borders from the specified image by the specified padding amount.
	//////////////////////////////////////////////////////////////////////////////////
	void removePadding(const Mat& sourceImage, Mat& destinationImage, const int padAmount)
	{
		sourceImage(Rect(padAmount, padAmount, sourceImage.size().width - 1 - padAmount, sourceImage.size().height - 1 - padAmount)).copyTo(destinationImage);
	}
}
