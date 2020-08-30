
/*
This is an adaption of the XorShift256 random generator
http://prng.di.unimi.it/xoshiro256plusplus.c

with the ability be seeded by SplitMix64
http://prng.di.unimi.it/splitmix64.c
*/

#include <iostream>
#include "Qpix/Random_Generator.h"


namespace Qpix
{

    uint64_t Random_Generator::rotl(const uint64_t x, int k) 
    {
        return (x << k) | (x >> (64 - k));
    }


    uint64_t Random_Generator::SplitMix64_next(void) 
    {
        uint64_t z = (seed_ += 0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

    void Random_Generator::set_seed(uint64_t seed)
    {
        seed_ = seed;
        s[0] = SplitMix64_next();
        s[1] = SplitMix64_next();
        s[2] = SplitMix64_next();
        s[3] = SplitMix64_next();
    }

    void Random_Generator::set_default()
    {
        s[0] = 23478234234;
        s[1] = 2342342345;
        s[2] = 234234121323;
        s[3] = 2234453453;
    }

    void Random_Generator::print_seed()
    {
        std::cout << s[0] << std::endl;
        std::cout << s[1] << std::endl;
        std::cout << s[2] << std::endl;
        std::cout << s[3] << std::endl;
    }

    double Random_Generator::XorShift256_next(void) 
    {
        const uint64_t result = ((s[0] + s[3])&0x000FFFFFFFFFFFFF)|0x3FF0000000000000;
        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 45);
        double a;
        memcpy(&a,&result,8);
        a-=1;
        return a;
    }


    /* This is the jump function for the generator. It is equivalent
    to 2^128 calls to next(); it can be used to generate 2^128
    non-overlapping subsequences for parallel computations. */
    void Random_Generator::jump(void) 
    {
        static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        uint64_t s2 = 0;
        uint64_t s3 = 0;
        for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
            for(int b = 0; b < 64; b++) 
            {
                if (JUMP[i] & UINT64_C(1) << b) 
                {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                XorShift256_next();	
            }
            
        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }


    /* This is the long-jump function for the generator. It is equivalent to
    2^192 calls to next(); it can be used to generate 2^64 starting points,
    from each of which jump() will generate 2^64 non-overlapping
    subsequences for parallel distributed computations. */
    void Random_Generator::long_jump(void) 
    {
        static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        uint64_t s2 = 0;
        uint64_t s3 = 0;
        for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
            for(int b = 0; b < 64; b++) 
            {
                if (LONG_JUMP[i] & UINT64_C(1) << b) 
                {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                XorShift256_next();	
            }
            
        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }



}