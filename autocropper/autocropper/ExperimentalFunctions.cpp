#include "ExperimentalFunctions.h"
#include "FileUtilities.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace experimental;
using namespace std;

namespace experimental
{
	//////////////////////////////////////////////////////////////////////////////////
	// and()
	//
	// Combine the images via an and operation -- useful in visualization of the
	// maximum extents of the root system.
	//////////////////////////////////////////////////////////////////////////////////
	Mat and(vector<Mat>& images)
	{
		if (images.size() == 0)
			return Mat();

		Mat andImage = images.at(0).clone();

		for (int i = 0; i < images.size(); ++i)
		{
			bitwise_and(andImage, images.at(i), andImage);
		}

		return andImage;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// computeForegroundMask()
	//
	// Computes the foreground mask based on some background subtraction method.
	//////////////////////////////////////////////////////////////////////////////////
	Mat computeForegroundMask(const vector<Mat>& images)
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

			//string filename = utility::FileUtilities::buildFilename("C:\\Temp\\images\\fg", ++i);
			//imwrite(filename, foregroundImage);
		}

		return foregroundImage;

		waitKey();
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

	//////////////////////////////////////////////////////////////////////////////////
	// generateEnhancedCenterMask()
	//
	// Generate a mask which has a value of 1.0 in the center, and 0.0 along the
	// edges, with a smooth gradient between.
	//////////////////////////////////////////////////////////////////////////////////
	Mat generateEnhancedCenterMask(Size size)
	{
		Mat image = Mat::ones(size, CV_8UC1);
		padImage(image, image, 1);
		distanceTransform(image, image, CV_DIST_C, 3);
		double maxVal;
		minMaxLoc(image, NULL, &maxVal);
		image *= 1/maxVal;

		removePadding(image, image, 1);

		return image;
	}
}
