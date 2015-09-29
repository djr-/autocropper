#include "experimental_functions.h"
#include "file_utilities.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

using namespace cv;
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
}