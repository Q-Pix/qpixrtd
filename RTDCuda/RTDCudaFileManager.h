#include "ROOTFileManager.h"


namespace Qpix {

    // sublcass the ROOTFileManager
    class RTDCudaFileManager : public ROOTFileManager
    {
        public:
            // allow inheritance of ROOTFileManager's constructors for ease of use.
            using ROOTFileManager::ROOTFileManager;

            // we're overloading the main function class to GetEvent() / Pixelize_Event() / and Reset_Fast() 
            std::vector<Qpix::ION> Get_Event(int);
            void AddEvent(std::vector<Qpix::Pixel_Info> const);
            void AddEvent(const std::set<int>&, std::unordered_map<int, Qpix::Pixel_Info>&);
            int GetCurrentEntry() const {return _currentEntry;};

        private:

            Qpix::Qpix_Paramaters _params;
            int _currentEntry = 0;

    };

}
