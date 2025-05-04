#include "host_utils.cuh"

void dbg_log(const char *msg){
    #if DEBUG
    printf("%s",msg);
    #endif
}



void startTimer(timeval_t *start_tv)
{
   gettimeofday((struct timeval*)start_tv, NULL);
}


void endTimer(const char *msg, timeval_t *start_tv)
{
   struct timeval end_tv;

   gettimeofday(&end_tv, NULL);

   time_t sec = end_tv.tv_sec - start_tv->tv_sec;
   time_t ms = end_tv.tv_usec - start_tv->tv_usec;

   time_t diff = sec * 1000000 + ms;

   printf("%39s:\t\t%fms\n", msg, (double)((double)diff/1000.0));
}


size_t readFileToBuf(FILE *fp, char **buf){
    fseek(fp, 0, SEEK_END);
	long fileSize = ftell(fp) + 1;
	rewind(fp);
    char* filebuf = (char*)malloc(fileSize);
    size_t read_size = fread(filebuf, fileSize, 1, fp);
	if(read_size != 1){
		// printf("Error reading file. Exiting %lu\n",read_size);
		// return 0;
	}
    fclose(fp);
    *buf = filebuf;
    return fileSize;
}

void stringToUpper(char *buf, size_t filesize){
    for(size_t i=0;i<filesize;i++){
            buf[i] = toupper(buf[i]);
    }
}



void printStringOfLen(char *buf, size_t start, size_t len){
    printf("start:%lu, length:%lu   ", start, len);
    for(size_t i=0;i<len;i++){
        printf("%c",buf[start+i]);
    }
    printf("\n");
}

void printString(char *buf, int start){
    for(int i=start;;i++){
        if(buf[i]=='\0'){
            printf(":");
            break;
        }
        printf("%c",buf[i]);
    }
}



inputRecordsList_t * newInputRecordsList(){
    inputRecordsList_t *l = (inputRecordsList_t *)malloc(sizeof(inputRecordsList_t));
    l->numRecords = 0;
    l->capacity = 0;
    l->list = NULL;
    return l;
}

void addInputRecord(inputRecordsList_t *inpRecList,size_t start, size_t len){
    const int allocNumRecords = 100000;
    const int allocChunkSize = sizeof(record_t) * allocNumRecords;

    if(inpRecList->capacity == 0){
        inpRecList->list = (record_t*)malloc(allocChunkSize);
        inpRecList->capacity += allocNumRecords;
    }

    if(inpRecList->numRecords == inpRecList->capacity){
        inpRecList->capacity += allocNumRecords;
        inpRecList->list = (record_t*)realloc(inpRecList->list, inpRecList->capacity * sizeof(record_t));
    }

    record_t *ptr = inpRecList->list;
    ptr += inpRecList->numRecords;

    ptr->recordStart = start;
    ptr->recordLen = len;
    inpRecList->numRecords++;
}



void splitFileBlocks(char *filebuf, size_t filesize, inputRecordsList_t *l){
    int offset = 0;
	char* p = filebuf;
	char* start = filebuf;
	while (1)
	{
		int blockSize = 2048;
		if (offset + blockSize > filesize) blockSize = filesize - offset;
		p += blockSize;
		for (; *p >= 'A' && *p <= 'Z'; p++);
			
		if (*p != '\0') 
		{
			*p = '\0'; 
			++p;
			blockSize = (int)(p - start);
            addInputRecord(l,offset, blockSize);
			offset += blockSize;
			start = p;
		}
		else
		{
			*p = '\0'; 
			blockSize = (int)(filesize - offset);
            addInputRecord(l,offset, blockSize);
			break;
		}
	}
}

