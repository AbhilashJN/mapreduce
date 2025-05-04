#include <stdio.h>
#include "mapreduce.cuh"
#include "host_utils.cuh"


int main(int argc, char** argv)
{
    struct timeval time_start, overall_time_start;

    if (argc != 2)
	{
		printf("usage: %s datafile\n", argv[0]);
		exit(-1);	
	}

    startTimer(&overall_time_start);
   
    FILE* fp = fopen(argv[1], "r");
    char *filebuf;
    size_t filesize = readFileToBuf(fp,&filebuf);
    if(filesize==0){
        return -1;
    }

    startTimer(&time_start);
    stringToUpper(filebuf, filesize);
    endTimer("convert to upper case [CPU]",&time_start);


    inputRecordsList_t *l = newInputRecordsList();

    startTimer(&time_start);
    splitFileBlocks(filebuf, filesize, l);
    endTimer("split file blocks [CPU]",&time_start);


    runMapReduce(filebuf,filesize, l);

    endTimer("overall time",&overall_time_start);

    return 0;
}