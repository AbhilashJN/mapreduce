#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H

#include<stdio.h>
#include<stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

// functor to compare strings in flat array based on offset;
struct StringKeyCompare {
    char *allKeys;

    __device__ bool operator()(int lhsOffset, int rhsOffset) {
        char *l = allKeys + lhsOffset;
        char *r = allKeys + rhsOffset;

        for( ; *l && *r && *l==*r; ){
            ++l;
            ++r;
        }

        return *l < *r;
    }
};


// functor to check quality of strings in flat array based on offset;
struct StringKeyEqual{
    char *allKeys;

    __device__ bool operator()(int lhsOffset, int rhsOffset) {
        char *l = allKeys + lhsOffset;
        char *r = allKeys + rhsOffset;

        for( ; *l && *r && *l==*r; ){
            ++l;
            ++r;
        }

        return *l == *r;
    }
};


__device__ void d_printBlock(char *filebuf, size_t start, size_t len);


void doPrefixSum(int *d_ptr,size_t numItems);

#endif