#ifndef MYTYPES_H
#define MYTYPES_H

#include<stdlib.h>

struct inputRecord {
    size_t recordStart;
    size_t recordLen;
};

typedef struct inputRecord record_t;

struct inputRecordsList {
    size_t numRecords;
    size_t capacity;
    record_t *list;
};


typedef struct inputRecordsList inputRecordsList_t;

#endif