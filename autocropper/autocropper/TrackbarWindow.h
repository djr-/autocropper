#pragma once

#include <opencv2/core.hpp>

namespace utility
{
	//////////////////////////////////////////////////////////////////////////////////
	// TrackbarWindow
	//
	// Encapsulate work required to create an OpenCV window with a trackbar.
	//////////////////////////////////////////////////////////////////////////////////
	class TrackbarWindow final
	{
	public:
		//TrackbarWindow(const std::string& windowName);
		TrackbarWindow(const std::string& windowName, const std::string& trackbarName, const int sliderDefault, const int sliderMax, cv::Mat(*trackbarBody)(cv::Mat image, int sliderValue));

		void show(const cv::Mat& image);
	private:
		static void onTrackbar(int trackbarPosition, void* userData);
		cv::Mat(*_trackbarBody)(cv::Mat image, int sliderValue);

		std::string _windowName;
		std::string _trackbarName;
		int _sliderValue;
		int _sliderMax;
		cv::Mat _image;
	};
}