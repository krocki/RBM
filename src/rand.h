/*
* @File: rand.h
* @Author: kamilrocki
* @Email: kmrocki@us.ibm.com
* @Created: 2015-04-29 10:43:55
* @Last Modified by:   kamilrocki
* @Last Modified time: 2015-04-29 20:32:14
*/

#ifndef __RAND_H__
#define __RAND_H__

#include <sys/time.h>
#include <random>

std::random_device rd;
std::mt19937_64 rng_mt19937_64(rd());
std::uniform_real_distribution<> u(0, 1);
std::uniform_int_distribution<> i(0, RAND_MAX);
std::normal_distribution<> n(0,1);

struct MT19937 {

    public:

        // This is equivalent to rand().
        static uint64_t randint() {
            return i(rng_mt19937_64);
        }

        static double rand(double min = 0, double max = 1) {
            return (u(rng_mt19937_64) + min)/(max - min);
        }

        static double randn(double mean = 0, double variance = 1) {
            return n(rng_mt19937_64) * variance + mean;
        }

};

#endif /*__RAND_H__*/