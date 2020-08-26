#ifndef READG4ROOT_H_
#define READG4ROOT_H_

// C++ includes
#include <iostream>
#include <string>
#include <vector>

// ROOT includes
#include "TBranch.h"
#include "TChain.h"
#include "TFile.h"
#include "TROOT.h"

#include "Qpix/Random.h"
#include "Qpix/Structures.h"


namespace Qpix
{
    class READ_G4_ROOT
    {
        private:
        //------------------------------------------------------------
        // set up branch variables
        //------------------------------------------------------------

        int run_;
        int event_;
        int number_particles_;
        int number_hits_;
        double energy_deposit_;

        std::vector< int >    * particle_track_id_;
        std::vector< int >    * particle_parent_track_id_;
        std::vector< int >    * particle_pdg_code_;
        std::vector< double > * particle_mass_;
        std::vector< double > * particle_charge_;
        std::vector< int >    * particle_process_key_;
        std::vector< int >    * particle_total_occupancy_;

        std::vector< double > * particle_initial_x_;
        std::vector< double > * particle_initial_y_;
        std::vector< double > * particle_initial_z_;
        std::vector< double > * particle_initial_t_;

        std::vector< double > * particle_initial_px_;
        std::vector< double > * particle_initial_py_;
        std::vector< double > * particle_initial_pz_;
        std::vector< double > * particle_initial_energy_;

        std::vector< int >    * hit_track_id_;
        std::vector< double > * hit_start_x_;
        std::vector< double > * hit_start_y_;
        std::vector< double > * hit_start_z_;
        std::vector< double > * hit_start_t_;
        std::vector< double > * hit_end_x_;
        std::vector< double > * hit_end_y_;
        std::vector< double > * hit_end_z_;
        std::vector< double > * hit_end_t_;
        std::vector< double > * hit_length_;
        std::vector< double > * hit_energy_deposit_;
        std::vector< int >    * hit_process_key_;

        // set branch addresses
        void set_branch_addresses(TChain * chain);

        // set the G4 tree
        TChain * chain_ = new TChain("g4_event_tree");

        public:

        // void Open_File( std::vector< std::string > file_list_ );
        void Open_File( std::string file_ );
        void Get_Event(int EVENT, Qpix::Liquid_Argon_Paramaters * LAr_params, std::vector<Qpix::ELECTRON>& hit_e);
        
    };//READ_G4_ROOT

}




#endif