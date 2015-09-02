#include "file_utilities.h"
#include <sys/stat.h>
#include <algorithm>
#include <sstream>

using namespace std;
using namespace utility;

//////////////////////////////////////////////////////////////////////////////////
// fileExists()
//
// Returns true if the specified file exists, false otherwise.
//////////////////////////////////////////////////////////////////////////////////
bool FileUtilities::fileExists(const string& fileName)
{
	struct stat buffer;
	return stat(fileName.c_str(), &buffer) == 0;
}

//////////////////////////////////////////////////////////////////////////////////
// buildFilename()
//
// Write a file of the format [prefix][number].[filetype].
// This function is intended for use in writing a series of debugging images.
//////////////////////////////////////////////////////////////////////////////////
string FileUtilities::buildFilename(const string& prefix, int number, const string& filetype)
{
	stringstream ss;
	ss << prefix << number << "." << filetype;
	return ss.str();
}
