#include "RTDCudaFileManager.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include "RTDCuda.h"

namespace Qpix {

    std::vector<Qpix::ION> RTDCudaFileManager::Get_Event(int evt)
    {
        std::vector<Qpix::ION> electrons;

        // continue reading from the TTree until Entry is not what we expect
        int maxEntries = in_ttree_->GetEntries();
        in_ttree_->GetEntry(_currentEntry++);
        // std::cout << "getting entry: " << _currentEntry;
        // std::cout << ", at event: " << event_ << ".. ";
        int total_e = 0;

        std::vector<double> v_hit_start_x, v_hit_start_y, v_hit_start_z, v_hit_start_t;
        std::vector<double> v_hit_step_x, v_hit_step_y, v_hit_step_z, v_hit_step_t;
        std::vector<int> v_hit_n;

        while(event_ == evt and _currentEntry < maxEntries){

            int Nelectron = 0;

            double energy_deposit = hit_energy_deposit_;  // MeV
            double length_of_hit = hit_length_;  // cm

            // Set up the paramaters for the recombiataion 
            double dEdx = energy_deposit/length_of_hit;
            double Recombonation = Modified_Box(dEdx);

            // to account for recombination or not
            // calcualte the number of electrons in the hit
            if (_params.Recombination)
            {
                Nelectron = round(Recombonation * (energy_deposit*1e6/_params.Wvalue));
            }else
            {
                Nelectron = round((energy_deposit*1e6/_params.Wvalue));
            }

            total_e += Nelectron;
            if(Nelectron > 0) {

                // from PreStepPoint
                v_hit_start_x.push_back(hit_start_x_);      // cm
                v_hit_start_y.push_back(hit_start_y_);      // cm
                v_hit_start_z.push_back(hit_start_z_);      // cm
                v_hit_start_t.push_back(hit_start_t_*1e-9); // sec

                // // from PostStepPoint
                // double end_x = hit_end_x_;      // cm
                // double end_y = hit_end_y_;      // cm
                // double end_z = hit_end_z_;      // cm
                // double end_t = hit_end_t_*1e-9; // sec

                // Determin the "step" size (pre to post hit)
                v_hit_step_x.push_back((hit_end_x_ - v_hit_start_x.back()) / Nelectron);
                v_hit_step_y.push_back((hit_end_y_- v_hit_start_y.back()) / Nelectron);
                v_hit_step_z.push_back((hit_end_z_ - v_hit_start_z.back()) / Nelectron);
                v_hit_step_t.push_back((hit_end_t_*1e-9 - v_hit_start_t.back()) / Nelectron);

                v_hit_n.push_back(total_e);
            }
            // load next event, and check if event number matches
            in_ttree_->GetEntry(_currentEntry++);
        }

        std::vector<Qpix::ION> h_ions(total_e);
        launch_add_diff_arrays(v_hit_start_x.data(), v_hit_step_x.data(), 
                               v_hit_start_y.data(), v_hit_step_y.data(), 
                               v_hit_start_z.data(), v_hit_step_z.data(), 
                               v_hit_start_t.data(), v_hit_step_t.data(), 
                               h_ions.data(), v_hit_n.data(), total_e, v_hit_n.size());

        if(total_e)
            std::cout << "found n ions: " << h_ions.size() << ", pos: (" << h_ions[0].x << ","
                                                                         << h_ions[0].y << ","
                                                                         << h_ions[0].z << ","
                                                                         << h_ions[0].t << ")\n";

        return h_ions;
    };

    void RTDCudaFileManager::AddEvent(std::vector<Qpix::Pixel_Info> const)
    {
        std::cout << "basic event added.\n";
    };

    void RTDCudaFileManager::AddEvent(const std::set<int>&, std::unordered_map<int, Qpix::Pixel_Info>&)
    {
        std::cout << "event added.\n";
    };
}
