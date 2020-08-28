// C++ includes
#include <iostream>
#include <string>
#include <vector>

// ROOT includes
#include "TBranch.h"
#include "TChain.h"
#include "TFile.h"
#include "TROOT.h"

#include "Qpix/Structures.h"
#include "Qpix/WriteRoot.h"

namespace Qpix
{
    // Setup the tree to be written 
    void Root_Writer::Book(std::string const file_path)
    {
        // ROOT output file
        tfile_ = new TFile(file_path.data(), "recreate", "q_pix_geant");

        // RTD event tree
        ttree_ = new TTree("RTD_event_tree", "RTD event tree");
        //ttree_->Branch("run",   &run_,   "run/I");
        ttree_->Branch("event", &event_, "event/I");
        ttree_->Branch("x", &x_);
        ttree_->Branch("y", &y_);
        ttree_->Branch("reset", &reset_);
    }//Book

    // Reset event variables after filling TTree objects per event
    void Root_Writer::EventReset()
    {   
        event_ = -1;
        x_.clear();
        y_.clear();
        reset_.clear();
    }// EventReset

    // Sets the event number
    void Root_Writer::SetEvent(int const value)
    {
        event_ = value;
    }//SetEvent
    
    // Adds event that needs to be filled
    void Root_Writer::AddEvent(const std::vector<Qpix::Pixel_Info> Pixel)
    {

        for (unsigned int i=0; i<Pixel.size() ; i++)
        {
            // vector of resets as double instead of int for workaround
            std::vector< double > reset_double;

            for (unsigned int j=0; j<Pixel[i].RESET.size(); j++)
            {
                // cast from int to double
                reset_double.push_back(static_cast<double>(Pixel[i].RESET[j]));
            }

            // add to tree vectors
            x_.push_back(Pixel[i].X_Pix);
            y_.push_back(Pixel[i].Y_Pix);
            reset_.push_back(reset_double);
        }

    }//AddEvent

    // fill TTree objects per event
    void Root_Writer::EventFill()
    {
        ttree_->Fill();
    }//EventFill

    // write TTree objects to file and close file
    void Root_Writer::Save()
    {
        tfile_->Write();
        tfile_->Close();
    }//Save

    // Replacate the MC file into the analysis file for safty
    void Root_Writer::Backfill( std::string file_ )
    {
        // get G4 and MARLEY trees from ROOT file
        TFile read_file(file_.data());

        TTree * read_g4_event_tree;
        TTree * read_marley_event_tree;

        read_file.GetObject("g4_event_tree", read_g4_event_tree);
        read_file.GetObject("marley_event_tree", read_marley_event_tree);

        // go into output file
        tfile_->cd();

        // clone G4 and MARLEY trees
        auto write_g4_event_tree = read_g4_event_tree->CloneTree();
        auto write_marley_event_tree = read_marley_event_tree->CloneTree();

        if (write_g4_event_tree == 0 or write_marley_event_tree == 0)
        {
            // maybe throw an exception here
            std::cout << "Could not copy trees to output file!" << std::endl;
        }
    }//Backfill


}

