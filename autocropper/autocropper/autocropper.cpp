#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat originalImage = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	imshow("Input image", originalImage);

	waitKey();

	return 0;
}
