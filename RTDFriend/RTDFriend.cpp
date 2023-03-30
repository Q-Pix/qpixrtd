// cpp
#include <string>
#include <vector>
#include <iostream>

// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"

// lib
#include "ROOTFileManager.h"

// iterate all through th// take an arbitrary list of input files with the last file being the output destination file
// create a TChain from all of the inputs, and fill the entry that has the lowest current time
int main(int argc, char** argv)
{
    if(argc != 2){
        std::cout << "friend error: incorrect number of input files!\n";
        return -1;
    }
    std::string input_file = argv[1];

    // take the rtd file with the metatree and eventtree, rip the relavant
    // metadata branches into the rtd branches
    TFile* tf = new TFile(input_file.c_str());
    if(tf->IsZombie()){std::cout << "RTDFriend WARNING: unable to open file!\n";};
    
    // make sure the trees that we want are there
    TTree* event_tree = (TTree*)tf->Get("event_tree");
    if(event_tree == NULL){std::cout << "WARNING unable to find event tree!\n"; return -1;};
    TTree* metadata = (TTree*)tf->Get("metadata");
    if(metadata == NULL){std::cout << "WARNING unable to find meta tree!\n"; return -1;};

    // file exists, let's try updating it
    delete tf;
    tf = new TFile(input_file.c_str(), "UPDATE");
    event_tree = (TTree*)tf->Get("event_tree");
    metadata = (TTree*)tf->Get("metadata");
    int nEntries = event_tree->GetEntries();

    // values
    int val = Qpix::RTDFriend(metadata, event_tree);
    if(val < 0){std::cout << "error in friend update!\n"; return -1;};
    
    tf->Write();
    std::cout << "tree updated!\n";

    // need to add meta params to these ROOT files before a hadd
    // so that a combined ROOT file can be created. The meta params allow for
    // dataframe filtering to select the desired event

    return 0;
}