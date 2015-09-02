#pragma once

#include <string>

namespace utility
{
	//////////////////////////////////////////////////////////////////////////////////
	// FileUtilities
	//
	// Helper functions for common file operations.
	//////////////////////////////////////////////////////////////////////////////////
	class FileUtilities final
	{
	public:
		static bool fileExists(const std::string& fileName);
		static std::string buildFilename(const std::string& prefix, int number, const std::string& filetype = "png");
	};
}
