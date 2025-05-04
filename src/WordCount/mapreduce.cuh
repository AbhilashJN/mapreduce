#ifndef MAP_CUH
#define MAP_CUH

#include<stdio.h>
#include<stdlib.h>
#include "mytypes.cuh"



int runMapReduce(char *inputFile, size_t filesize, inputRecordsList_t *inpRecList);
#endif