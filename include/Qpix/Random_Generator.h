#ifndef RANDOM_GENERATOR_H_
#define RANDOM_GENERATOR_H_

#include <cstdint>

namespace Qpix
{
    class Random_Generator
    {
    private:
        uint64_t s[4]={23478234234,2342342345,234234121323,2234453453};

        uint64_t seed_ = 777; 

    public:

        static inline uint64_t rotl(const uint64_t x, int k);

        uint64_t SplitMix64_next(void);

        void set_seed(uint64_t seed);

        void set_default();

        void print_seed();

        double XorShift256_next(void);
        
        void jump(void);
        
        void long_jump(void);
    };
}
#endif