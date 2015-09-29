#include "experimental_functions.h"

using namespace cv;
using namespace std;

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
