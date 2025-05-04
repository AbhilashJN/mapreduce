#ifndef HOST_UTILS_H
#define HOST_UTILS_H

#include<stdlib.h>
#include<stdio.h>
#include <ctype.h>
#include <sys/time.h>
#include "mytypes.cuh"

typedef struct timeval timeval_t;



#define DEBUG 1

void dbg_log(const char *msg);

void startTimer(timeval_t *start_tv);


void endTimer(const char *msg, timeval_t *start_tv);

size_t readFileToBuf(FILE *fp, char **buf);

void stringToUpper(char *buf, size_t filesize);


void printStringOfLen(char *buf, size_t start, size_t len);

void printString(char *buf, int start);



inputRecordsList_t * newInputRecordsList(void);

void addInputRecord(inputRecordsList_t *inpRecList,size_t start, size_t len);


void splitFileBlocks(char *filebuf, size_t filesize, inputRecordsList_t *l);
#endif