// C++ includes
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <getopt.h>

// Qpix includes
#include "Random.h"
#include "ROOTFileManager.h"
#include "Structures.h"
#include "PixelResponse.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TObject.h"
#include "Rtypes.h"



//----------------------------------------------------------------------
// declare global variables
//----------------------------------------------------------------------

static int f_noise;
static int f_reco;
static int threshold = 0;
static std::string file_in;
static std::string file_out;


//----------------------------------------------------------------------
// main function
//----------------------------------------------------------------------
int main(int argc, char** argv)
{

  int c;

  while (1)
  {
    int option_index = 0;
    static struct option long_options[] =
    {
      {"nonoise",           no_argument,        &f_noise,     1},
      {"recombination",   no_argument,        &f_reco,      1},
      {"threshold",       required_argument,  NULL,       't'},
      {"input",           required_argument,  NULL,       'i'},
      {"output",          required_argument,  NULL,       'o'},
      {NULL,              0,                  NULL,         0}
    };

    c = getopt_long(argc, argv, ":i:o:t:", long_options, &option_index);
    if (c == -1)
      break;

    switch (c)
    {
      case 'i':
        //printf("Option i has arg: %s\n", optarg);
        file_in = optarg;
        break;
      case 'o':
        //printf("Option o has arg: %s\n", optarg);
        file_out = optarg;
        break;
      case 't':
        //printf("Option t has arg: %s\n", optarg);
        threshold = atoi(optarg);
        break;
      case '?':
        printf("Unknown option: %c\n", optopt);
        break;
      case ':':
        printf("Missing arg for %c\n", optopt);
        break;
      case 1:
        printf("Non-option arg: %s\n", optarg);
        break;
    }
  }

  clock_t time_req;
  time_req = clock();
  double time;

  // changing the seed for the random numbergenerator and generating the noise vector 
  constexpr std::uint64_t Seed = 777;
  Qpix::Random_Set_Seed(Seed);
  std::vector<double> Gaussian_Noise(1e7,0.0);// = Qpix::Make_Gaussian_Noise(0, (int) 1e7);
  if (f_noise == 1)
	Gaussian_Noise = Qpix::Make_Gaussian_Noise(0, (int) 1e7);


  // In and out files
  //std::string file_in = argv[1];
  //std::string file_out = argv[2];

  // Qpix paramaters 
  Qpix::Qpix_Paramaters * Qpix_params = new Qpix::Qpix_Paramaters();
  set_Qpix_Paramaters(Qpix_params);
  Qpix_params->Buffer_time = 0.01;
  Qpix_params->Recombination = false;
  if (f_reco == 1)
	Qpix_params->Recombination = true;
  Qpix_params->Reset = threshold;
  print_Qpix_Paramaters(Qpix_params);

  // root file manager
  int number_entries = -1;
  std::cout << "test point 1" << std::endl;
  Qpix::ROOTFileManager rfm(file_in, file_out);
  std::cout << "test point 2" << std::endl;
  rfm.AddMetadata(Qpix_params);  // add parameters to metadata
  std::cout << "test point 3" << std::endl;
  number_entries = rfm.NumberEntries();
  std::cout << "Number Entries: " << number_entries << std::endl;
  rfm.EventReset();

  // Loop though the events in the file
  for (int evt = 0; evt < number_entries; evt++)
  {
    std::cout << "*********************************************" << std::endl;
    std::cout << "Starting on event " << evt << std::endl;

    std::cout << "Getting the event" << std::endl;
    std::vector<Qpix::ELECTRON> hit_e;
    // turn the Geant4 hits into electrons
    rfm.Get_Event( evt, Qpix_params, hit_e);
    std::cout << "size of hit_e = " << hit_e.size() << std::endl;

    std::cout << "Pixelizing the event" << std::endl;
    Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
    std::vector<Qpix::Pixel_Info> Pixel;
    // Pixelize the electrons 
    PixFunc.Pixelize_Event( hit_e, Pixel );

    std::cout << "Running the resets" << std::endl;
    // the reset function
    // PixFunc.Reset(Qpix_params, Gaussian_Noise, Pixel);
    PixFunc.Reset_Fast(Qpix_params, Gaussian_Noise, Pixel);

    rfm.AddEvent( Pixel );
    rfm.EventFill();
    rfm.EventReset();
  }

  // save and close
  rfm.Save();




  std::cout << "done" << std::endl;
  time_req = clock() - time_req;
  time = (float)time_req/CLOCKS_PER_SEC;
  std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
  return 0;

} // main()
