// cpp
#include <string>
#include <vector>
#include <iostream>
#include <map>

// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TInterpreter.h"

#include "rtdRadiogenic.h"

int main(int argc, char** argv)
{
    if(argc != 2){
        std::cout << "friend error: incorrect number of input files!\n";
        return -1;
    }
    std::string input_file = argv[1];
    // make sure that we can read in a map for all of the pixels
    std::unordered_map<int, reset_data> pixel_resets;
    for(int xi=1; xi<576; ++xi){
        for(int yi=1; yi<1501; ++yi)
            pixel_resets[encode(xi, yi)] = reset_data();
    }

    // take the rtd file with the metatree and eventtree, rip the relavant
    // metadata branches into the rtd branches
    TFile* tf = new TFile(input_file.c_str(), "UPDATE");
    if(tf->IsZombie()){std::cout << "rtd_Radiogenic WARNING: unable to open file!\n";};

    int pixel_x, pixel_y;
    double pixel_reset;
    std::vector<int>* pixel_reset_truth_track_id = 0;
    std::vector<int>* pixel_reset_truth_weight = 0;

    TTree* event_tree = (TTree*)tf->Get("event_tree");
    // make sure the trees that we want are there
    if(event_tree == NULL){std::cout << "WARNING unable to find event tree!\n"; return -1;};
    event_tree->SetBranchAddress("pixel_x", &pixel_x);
    event_tree->SetBranchAddress("pixel_y", &pixel_y);
    event_tree->SetBranchAddress("pixel_reset", &pixel_reset);
    event_tree->SetBranchAddress("pixel_reset_truth_track_id", &pixel_reset_truth_track_id);
    event_tree->SetBranchAddress("pixel_reset_truth_weight", &pixel_reset_truth_weight);

    // loop through the tree and build the map we want
    std::cout << "building entries..\n";
    for(int ei=0; ei<event_tree->GetEntries(); ++ei){
        event_tree->GetEntry(ei);
        pixel_resets[encode(pixel_x, pixel_y)].pixel_resets.push_back(pixel_reset);
        pixel_resets[encode(pixel_x, pixel_y)].pixel_reset_truth_track_id.push_back(*pixel_reset_truth_track_id);
        pixel_resets[encode(pixel_x, pixel_y)].pixel_reset_truth_weight.push_back(*pixel_reset_truth_weight);
    }

    std::vector<double>* p_pixel_resets = 0;
    std::vector<std::vector<int>>* p_pixel_track_ids = 0;
    std::vector<std::vector<int>>* p_pixel_weights = 0;

    int px, py;
    TTree* ptt = new TTree("pixel_event_data", "tt");
    ptt->Branch("pixel_x", &px);
    ptt->Branch("pixel_y", &py);
    ptt->Branch("pixel_reset", &p_pixel_resets);
    ptt->Branch("pixel_reset_truth_track_id", &p_pixel_track_ids);
    ptt->Branch("pixel_reset_truth_weight", &p_pixel_weights);

    // now loop through the map and fill the tree appropriately
    std::cout << "entries built. Filling tree\n";
    for(int xi=1; xi<576; ++xi){
        for(int yi=1; yi<1501; ++yi){
            px = xi;
            py = yi;
            p_pixel_resets = &pixel_resets[encode(xi, yi)].pixel_resets;
            p_pixel_track_ids = &pixel_resets[encode(xi, yi)].pixel_reset_truth_track_id;
            p_pixel_weights = &pixel_resets[encode(xi, yi)].pixel_reset_truth_weight;
            ptt->Fill();
            pixel_resets.erase(encode(xi, yi)); // be nice to the memory
        }
    }
    
    tf->Write();
    std::cout << "pixel tree created!\n";

    return 0;
}