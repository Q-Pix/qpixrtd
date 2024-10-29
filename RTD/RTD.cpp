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


//----------------------------------------------------------------------
// main function
//----------------------------------------------------------------------
int main(int argc, char** argv)
{

  static int f_noise;
  static int f_reco;
  static int f_twind;
  static int threshold = 0;
  static double pix_dim = 0;
  static std::string file_in;
  static std::string file_out;
  static double downsample = 1.;


  int c;

  while (1)
  {
    int option_index = 0;
    static struct option long_options[] =
    {
      /* these options set flags */
      {"nonoise",         no_argument,        &f_noise,     1},
      {"norecombination", no_argument,        &f_reco,      1},
      {"notimewindow",    no_argument,        &f_twind,     1},
      /* these options take inputs */
      {"threshold",       required_argument,  NULL,         't'},
      {"pix_dim",         required_argument,  NULL,         's'},
      {"input",           required_argument,  NULL,       'i'},
      {"output",          required_argument,  NULL,       'o'},
      {"downsample",      required_argument,  NULL,         'd'},
      {NULL,              0,                  NULL,         0}
      };

    c = getopt_long(argc, argv, ":i:o:t:s:d:", long_options, &option_index);
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
        printf("Option t has arg: %s\n", optarg);
        threshold = atoi(optarg);
        break;
      case 's':
        printf("Option s has arg: %s\n", optarg);
        pix_dim = atof(optarg);
        break;
      case 'd':
        printf("Option d has arg: %s\n", optarg);
        downsample = atof(optarg);
        if (downsample > 1.) {std::cout << "downsampling must be a fraction less than 1" <<std::endl; abort;}
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


  // --------------------------------------
  // Setting Qpix paramaters 
  // --------------------------------------

  Qpix::Qpix_Paramaters * Qpix_params = new Qpix::Qpix_Paramaters();
  set_Qpix_Paramaters(Qpix_params);

  // Set Qpix Buffer time
  Qpix_params->Buffer_time = 0.01;

  // Set Recombination (if --norecombination is passed, Recombination will be turned off)
  if (f_reco == 1) Qpix_params->Recombination = false;

  // Set Noise (if --nonoise is passed, Noise will be turned off)
  if (f_noise == 1) Qpix_params->Noise = false;

  // Set Time Window Flag
  if (f_twind == 1) Qpix_params->TimeWindow = false;


  // Set Reset threshold to threshold passed
  Qpix_params->Reset = threshold;

  // Set Pixel Size
  Qpix_params->Pix_Size = pix_dim;

  // Set Downsampling (will be set to 1 by default)
  Qpix_params->Sampling = downsample;

  // Print the Qpix_params for the run
  print_Qpix_Paramaters(Qpix_params);

  
  // ----------------------------------------
  // Generating Noise vector
  // ----------------------------------------

  // changing the seed for the random numbergenerator and generating the noise vector 
  constexpr std::uint64_t Seed = 777;
  Qpix::Random_Set_Seed(Seed);
  // Initialize empty vector (synonymous with no noise)
  // If Noise is turned on, fill empty noise vector with Gaussian noise
  std::vector<double> Gaussian_Noise(1e7,0.0);
  if (Qpix_params->Noise == true) {Gaussian_Noise = Qpix::Make_Gaussian_Noise(0, (int) 1e7);}

  // -----------------------------------------
  // root file manager
  // -----------------------------------------
  int number_entries = -1;
  Qpix::ROOTFileManager rfm = Qpix::ROOTFileManager(file_in, file_out);
  rfm.AddMetadata(Qpix_params);  // add parameters to metadata
  number_entries = rfm.NumberEntries();
  rfm.EventReset();

  // -------------------------------------------
  // Loop though the events in the file
  // -------------------------------------------
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
