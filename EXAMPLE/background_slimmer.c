// -----------------------------------------------------------------------------
//  background_slimmer.c
//
//  root -l -b -q 'background_slimmer.c("/path/to/input.root", "/path/to/output.root")'
//  https://root-forum.cern.ch/t/delete-multiple-branches-from-atootfile-at-once/41809/5
//   * Author: Everybody is an author!
//   * Creation date: 17 August 2021
// -----------------------------------------------------------------------------

// C++ includes
#include <iostream>
#include <string>
#include <vector>

// ROOT includes
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"

//----------------------------------------------------------------------
// run, forrest, run!
//----------------------------------------------------------------------
void background_slimmer(std::string input, std::string output)
{
    std::cout << "input: " << input << std::endl;
    std::cout << "output: " << output << std::endl;

    // open input file
    TFile * input_file = new TFile(input.data(), "read");

    // get TTree objects
    TTree * input_metadata;
    TTree * input_event_tree;

    input_file->GetObject("metadata", input_metadata);
    input_file->GetObject("event_tree", input_event_tree);

    // deactivate all branches
    // input_metadata->SetBranchStatus("*", 1);
    input_event_tree->SetBranchStatus("*", 0);

    std::vector< std::string > branch_names = {

        "run",
        "event",

        // "generator_initial_number_particles",
        // "generator_initial_particle_x",
        // "generator_initial_particle_y",
        // "generator_initial_particle_z",
        // "generator_initial_particle_t",
        // "generator_initial_particle_px",
        // "generator_initial_particle_py",
        // "generator_initial_particle_pz",
        // "generator_initial_particle_energy",
        // "generator_initial_particle_pdg_code",
        // "generator_initial_particle_mass",
        // "generator_initial_particle_charge",

        // "generator_final_number_particles",
        // "generator_final_particle_x",
        // "generator_final_particle_y",
        // "generator_final_particle_z",
        // "generator_final_particle_t",
        // "generator_final_particle_px",
        // "generator_final_particle_py",
        // "generator_final_particle_pz",
        // "generator_final_particle_energy",
        // "generator_final_particle_pdg_code",
        // "generator_final_particle_mass",
        // "generator_final_particle_charge",

        // "number_particles",
        // "number_hits",

        "energy_deposit",

        // "particle_track_id",
        // "particle_parent_track_id",
        // "particle_pdg_code",
        // "particle_mass",
        // "particle_charge",
        // "particle_process_key",
        // "particle_total_occupancy",
        // "particle_initial_x",
        // "particle_initial_y",
        // "particle_initial_z",
        // "particle_initial_t",
        // "particle_initial_px",
        // "particle_initial_py",
        // "particle_initial_pz",
        // "particle_initial_energy",

        // "particle_number_daughters",
        // "particle_daughter_track_id",

        "pixel_x",
        "pixel_y",
        "pixel_reset",
        "pixel_tslr",

        // "pixel_reset_truth_track_id",
        // "pixel_reset_truth_weight",

    };

    // activate select branches
    for (auto branch_name : branch_names)
    {
        input_event_tree->SetBranchStatus(branch_name.data(), 1);
    }

    // create a new file and clones of old trees in new file
    TFile * output_file = new TFile(output.data(), "recreate");
    TTree * output_event_tree = input_event_tree->CloneTree(-1, "fast");
    TTree * output_metadata = input_metadata->CloneTree(-1, "fast");

    output_metadata->Print();
    output_event_tree->Print();
    output_file->Write();
    
} // background_slimmer()

