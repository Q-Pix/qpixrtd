#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

namespace Qpix
{
    std::vector<double> convertStringVectortoDoubleVector(const std::vector<std::string>& stringVector)
    {
        std::vector<double> doubleVector(stringVector.size());
        std::transform(stringVector.begin(), stringVector.end(), doubleVector.begin(), [](const std::string& val)
        {
            return std::stod(val);
        });
        return doubleVector;
    }


    void DataFileParser2(std::string FILE, std::vector<std::vector<double>>& RawDataVector, std::vector<int>& EventLengths)
    {
        std::ifstream file(FILE);
        EventLengths.push_back(0);
        int count = 0;
        int runner = 0;

        while (!file.eof()) 
        {    
            std::vector<std::string> tmpVec;
            std::vector<double> tmpVecD;
            std::string tmpString;

            for (int j = 0; j < 5; j++)
            {
            file  >> tmpString;
            if(tmpString.empty())
            {
                EventLengths.push_back(runner);
                goto endloop;
            }
            tmpVec.push_back(tmpString);
            }
            tmpVecD = convertStringVectortoDoubleVector(tmpVec);
            if (count == tmpVecD[0]){runner+=1;}
            else{count+=1;  EventLengths.push_back(runner); runner+=1;}

            RawDataVector.push_back(tmpVecD);
        }
        endloop:
        return ;
    }
    

    std::vector<std::vector<double>> GetEventVector(int Event, std::vector<int> EventLengths, std::vector<std::vector<double>> data)
    {
        std::vector<std::vector<double>> event;
        int start = EventLengths[Event];
        int end   = EventLengths[Event+1];

        for (int i = start; i < end; i++) 
        {
            std::vector<double> tmpVec;
            for (int j = 1; j < data[i].size(); j++)
            {
            tmpVec.push_back(data[i][j]);
            }
            event.push_back(tmpVec);
        }
        return event;
    }

}