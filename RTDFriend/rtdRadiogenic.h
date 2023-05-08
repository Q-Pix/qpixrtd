#include <vector>

int encode(const int& px, const int& py){
    int id = py*10000 + px;
    return id;
}

struct reset_data{
    std::vector<double> pixel_resets;
    std::vector<std::vector<int>> pixel_reset_truth_track_id;
    std::vector<std::vector<int>> pixel_reset_truth_weight;
};
