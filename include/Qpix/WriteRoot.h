#ifndef WRITEROOT_H_
#define WRITEROOT_H_

// C++ includes
#include <iostream>
#include <string>
#include <vector>

// ROOT includes
#include "TBranch.h"
#include "TChain.h"
#include "TFile.h"
#include "TROOT.h"

// #include "Qpix/Random.h"
#include "Qpix/Structures.h"
// #include "Qpix/PixelResponse.h"

namespace Qpix
{
    class Root_Writer
    {
    private:
        // ROOT objects
        TFile * tfile_;
        TTree * ttree_;

        // variables that will go into the event trees
        // int run_;
        int event_;

        std::vector< int > x_;
        std::vector< int > y_;
        std::vector< std::vector < double > > reset_;

    public:

        void Book(std::string const file_path);

        void EventReset();

        void SetEvent(int const value);
        
        void AddEvent(const std::vector<Qpix::Pixel_Info> Pixel);

        void EventFill();

        void Save();

        void Backfill( std::string file_ );

    };

}


#endif