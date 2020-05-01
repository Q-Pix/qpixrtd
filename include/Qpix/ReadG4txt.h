#ifndef READG4TXT_H_
#define READG4TXT_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

namespace Qpix
{
    std::vector<double> convertStringVectortoDoubleVector(const std::vector<std::string>& stringVector);

    void DataFileParser2(std::string FILE, std::vector<std::vector<double>>& RawDataVector, std::vector<int>& EventLengths);

    std::vector<std::vector<double>> GetEventVector(int Event, std::vector<int> EventLengths, std::vector<std::vector<double>> data);

}

#endif