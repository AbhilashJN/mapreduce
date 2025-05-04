#include "mapreduce.cuh"
#include "device_utils.cuh"
#include "host_utils.cuh"


__global__ void CalculateMapPairsSize(
    record_t *inputRecords, int numRecords, char *file, 
    int *pairCountPerThread, int *totalKeysSizePerThread, int *totalValuesSizePerThread
){
    size_t tid =  blockDim.x * blockIdx.x + threadIdx.x;
    if(tid>=numRecords){
        return;
    }

    record_t rec = inputRecords[tid];

	char* ptrBuf = file + rec.recordStart;
	int line_size = rec.recordLen;

	char* p = ptrBuf;
	int lsize = 0;
	int wsize = 0;
	char* start = ptrBuf;

	while(1)
	{
		for (; *p >= 'A' && *p <= 'Z'; p++, lsize++);
		*p = '\0';
		++p;
		++lsize;
		wsize = (int)(p - start);
		if (wsize > 6)
		{
            pairCountPerThread[tid]++;
            totalKeysSizePerThread[tid]+=wsize;
            totalValuesSizePerThread[tid]+=sizeof(int);
            // printf("wordstart: %d, wordlen:%d \n", (int)(start-file), wsize);
		}
		for (; (lsize < line_size) && (*p < 'A' || *p > 'Z'); p++, lsize++);
		if (lsize >= line_size) break;
		start = p;
	}
}



__global__ void Map(
    record_t *inputRecords, int numRecords, char *file, 
    int *keyBaseOffsetPerThread, int *valBaseOffsetPerThread,
    char *d_keys, int *d_vals
){
    size_t tid =  blockDim.x * blockIdx.x + threadIdx.x;
    if(tid>=numRecords){
        return;
    }


    char *keyWriteLoc = d_keys + keyBaseOffsetPerThread[tid];
    int *valWriteLoc = d_vals + (valBaseOffsetPerThread[tid]/(sizeof(int)));
    
    record_t rec = inputRecords[tid];

	char* ptrBuf = file + rec.recordStart;
	int line_size = rec.recordLen;

	char* p = ptrBuf;
	int lsize = 0;
	int wsize = 0;
	char* start = ptrBuf;

	while(1)
	{
		for (; *p >= 'A' && *p <= 'Z'; p++, lsize++);
		*p = '\0';
		++p;
		++lsize;
		wsize = (int)(p - start);
		if (wsize > 6)
		{
            // d_keys[keyOffset]
            memcpy(keyWriteLoc, start, wsize);
            *valWriteLoc = (int)(keyWriteLoc - d_keys);

            keyWriteLoc += wsize;
            valWriteLoc += 1;
		}
		for (; (lsize < line_size) && (*p < 'A' || *p > 'Z'); p++, lsize++);
		if (lsize >= line_size) break;
		start = p;
	}
}


void getIntermediateSizes(
    int *totalPairCount, int *totalKeySize, int *totalValSize,
    int *d_pairCountPerThread, int *d_totalKeysSizePerThread, int *d_totalValuesSizePerThread,
    int numThreads
    ){
    int *lastPairCount = (int*)malloc(sizeof(int));
    int *lastKeySize = (int*)malloc(sizeof(int));
    int *lastValSize = (int*)malloc(sizeof(int));

    cudaMemcpy(lastPairCount, d_pairCountPerThread + numThreads - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(lastKeySize, d_totalKeysSizePerThread + numThreads - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(lastValSize, d_totalValuesSizePerThread + numThreads - 1, sizeof(int), cudaMemcpyDeviceToHost);

    doPrefixSum(d_pairCountPerThread, numThreads);
    doPrefixSum(d_totalKeysSizePerThread, numThreads);
    doPrefixSum(d_totalValuesSizePerThread, numThreads);
  
    cudaMemcpy(totalPairCount, d_pairCountPerThread + numThreads - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(totalKeySize, d_totalKeysSizePerThread + numThreads - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(totalValSize, d_totalValuesSizePerThread + numThreads - 1, sizeof(int), cudaMemcpyDeviceToHost);

    *totalPairCount = *totalPairCount + *lastPairCount;
    *totalKeySize = *totalKeySize + *lastKeySize;
    *totalValSize = *totalValSize + *lastValSize;
}



int copyInputToDevice(
    char *inputFile, size_t filesize, inputRecordsList_t *h_inpRecList, 
    char **d_inputFilePtr, record_t **d_inpRecordsPtr
){
    char *d_inputFile = NULL;
    record_t *d_inpRecords = NULL;

    cudaError_t err = cudaMalloc((void**)&d_inputFile, filesize);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n",cudaGetErrorString(err));
        return -1;
    }
    cudaMemcpy(d_inputFile, inputFile, filesize, cudaMemcpyHostToDevice);


    size_t recListSize = (h_inpRecList->numRecords * (sizeof(record_t)));
    err = cudaMalloc((void**)&d_inpRecords, recListSize);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n",cudaGetErrorString(err));
        return -1;
    }
    cudaMemcpy(d_inpRecords, h_inpRecList->list, recListSize, cudaMemcpyHostToDevice);

    *d_inputFilePtr = d_inputFile;
    *d_inpRecordsPtr = d_inpRecords;

    return 0;
}



int calculateMapOutputSize(
    char *d_inputFile, record_t *d_inputRecords, size_t numRecords,
    int numThreads, int blocksPerGrid, int threadsPerBlock,
    int **d_pairCountPerThreadPtr, int **d_totalKeysSizePerThreadPtr,  int **d_totalValuesSizePerThreadPtr
    ){
    int *d_pairCountPerThread=NULL;
    int *d_totalKeysSizePerThread=NULL;
    int *d_totalValuesSizePerThread=NULL;

    cudaError_t err= cudaMalloc((void**)&d_pairCountPerThread, numThreads*(sizeof(int)));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n",cudaGetErrorString(err));
        return -1;
    }


    err = cudaMalloc((void**)&d_totalKeysSizePerThread, numThreads*(sizeof(int)));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n",cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&d_totalValuesSizePerThread, numThreads*(sizeof(int)));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n",cudaGetErrorString(err));
        return -1;
    }
   
    CalculateMapPairsSize<<<blocksPerGrid, threadsPerBlock>>>(d_inputRecords, numRecords, d_inputFile, d_pairCountPerThread, d_totalKeysSizePerThread, d_totalValuesSizePerThread);
    cudaDeviceSynchronize();

    *d_pairCountPerThreadPtr = d_pairCountPerThread;
    *d_totalKeysSizePerThreadPtr = d_totalKeysSizePerThread;
    *d_totalValuesSizePerThreadPtr = d_totalValuesSizePerThread;
    return 0;
}


int runMap(
    int totalKeySize, int totalValSize, int blocksPerGrid, int threadsPerBlock,
    char *d_inputFile, record_t *d_inpRecords, int numRecords,
    int *d_totalKeysSizePerThread, int *d_totalValuesSizePerThread,
    char **d_keysPtr, int **d_valsPtr 
){
    char *d_keys = NULL;
    int *d_vals = NULL;

    cudaError_t err = cudaMalloc((void**)&d_keys, (size_t)(totalKeySize));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n",cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&d_vals, (size_t)(totalValSize));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n",cudaGetErrorString(err));
        return -1;
    }
    
    Map<<<blocksPerGrid, threadsPerBlock>>>(d_inpRecords, numRecords, d_inputFile, d_totalKeysSizePerThread, d_totalValuesSizePerThread, d_keys, d_vals);
    cudaDeviceSynchronize();

    *d_keysPtr = d_keys;
    *d_valsPtr = d_vals;
    return 0;
}

int sortKeyValuePairs(
    char *d_keys, int *d_vals, char *h_keys, int *h_vals,
    int totalPairCount, int totalValSize
){
    struct StringKeyCompare skCompare;
    skCompare.allKeys = d_keys;
    thrust::device_ptr<int> td_vals = thrust::device_pointer_cast(d_vals);

    thrust::sort(td_vals, td_vals + (totalPairCount), skCompare);
    cudaMemcpy(h_vals, d_vals, totalValSize, cudaMemcpyDeviceToHost);
    return 0;
}


int runReduceByKey(
    int totalPairCount, int totalOffsetsSize,
    char *d_allKeys, int *d_keyOffsets,
    int **d_uniqueWordOffsetsPtr,int **d_wordCountsPtr, int *numUniqueWordsPtr
){
    int *d_wordCounts = NULL;
    cudaError_t err = cudaMalloc((void**)&d_wordCounts, sizeof(int)*(totalPairCount));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n",cudaGetErrorString(err));
        return -1;
    }

    
    int *d_UniqueWordOffsets = NULL;
    err = cudaMalloc((void**)&d_UniqueWordOffsets, (size_t)(totalOffsetsSize));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n",cudaGetErrorString(err));
        return -1;
    }

    thrust::device_ptr<int> td_vals = thrust::device_pointer_cast(d_keyOffsets);
    thrust::device_ptr<int> td_wordCounts = thrust::device_pointer_cast(d_wordCounts);
    thrust::device_ptr<int> td_UniqueWordOffsets = thrust::device_pointer_cast(d_UniqueWordOffsets);
    thrust::constant_iterator<int> const_iter(1);
    
    struct StringKeyEqual skEqual;
    skEqual.allKeys = d_allKeys;

    int numUniqueWords = thrust::reduce_by_key(td_vals, td_vals+(totalPairCount) ,const_iter, td_UniqueWordOffsets, td_wordCounts,skEqual).first - td_UniqueWordOffsets;
    *d_uniqueWordOffsetsPtr = d_UniqueWordOffsets;
    *d_wordCountsPtr = d_wordCounts;
    *numUniqueWordsPtr = numUniqueWords;

    return 0;
}


void printWordWithCount(char *buf, int start,int count){
    for(int i=start;;i++){
        if(buf[i]=='\0'){
            printf(" | Wordsize: %d | Wordcount: %d\n", i+1-start, count);
            break;
        }
        printf("%c",buf[i]);
    }
}


void printOutput(char *h_allKeys, int *h_offsets, int *h_wordCounts, int numUniqueWords, int limit){
    printf("\n");
    printf("Number of unique words: %d\n\n",numUniqueWords);
    printf("First %d words (alphabetical order):\n\n",limit);
    for(int i=0;i<numUniqueWords && i<limit;i++){
        printWordWithCount(h_allKeys, h_offsets[i], h_wordCounts[i]);
    }
}



int runMapReduce(char *h_inputFile, size_t filesize, inputRecordsList_t *h_inpRecList){
    struct timeval time_start;
    
    char *d_inputFile = NULL;
    record_t *d_inpRecords = NULL;

    startTimer(&time_start);
    copyInputToDevice(h_inputFile, filesize, h_inpRecList, &d_inputFile, &d_inpRecords);
    endTimer("copy input to device [CPU->GPU]", &time_start);

    int threadsPerBlock = 256;
    int blocksPerGrid = (h_inpRecList->numRecords + threadsPerBlock - 1) / threadsPerBlock;
    int numThreads = blocksPerGrid * threadsPerBlock;

    int *d_pairCountPerThread = NULL;
    int *d_totalKeysSizePerThread = NULL;
    int *d_totalValuesSizePerThread = NULL;
  


    startTimer(&time_start);
    calculateMapOutputSize(
        d_inputFile, d_inpRecords, h_inpRecList->numRecords,
        numThreads, blocksPerGrid, threadsPerBlock,
        &d_pairCountPerThread, &d_totalKeysSizePerThread, &d_totalValuesSizePerThread
        );
    endTimer("calculate map output size [GPU]", &time_start);



    int totalPairCount=0;
    int totalKeySize=0;
    int totalValSize=0;
    startTimer(&time_start);
    getIntermediateSizes(&totalPairCount, &totalKeySize, &totalValSize, d_pairCountPerThread, d_totalKeysSizePerThread, d_totalValuesSizePerThread, numThreads);
    endTimer("calculate intermediate sizes [GPU]", &time_start);



    char *d_keys = NULL;
    int *d_vals = NULL;
    startTimer(&time_start);
    runMap(
        totalKeySize, totalValSize, blocksPerGrid, threadsPerBlock,
        d_inputFile, d_inpRecords, h_inpRecList->numRecords,
        d_totalKeysSizePerThread, d_totalValuesSizePerThread,
        &d_keys, &d_vals
    );
    endTimer("map [GPU]", &time_start);


    startTimer(&time_start);
    char *h_keys = (char*)malloc(totalKeySize);
    int *h_vals = (int*)malloc(totalValSize);
    cudaMemcpy(h_keys, d_keys, totalKeySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals, d_vals, totalValSize, cudaMemcpyDeviceToHost);
    endTimer("copy key-values to host [GPU->CPU]", &time_start);
    
    cudaFree(d_inputFile);
    cudaFree(d_inpRecords);

    startTimer(&time_start);
    sortKeyValuePairs(d_keys, d_vals, h_keys, h_vals, totalPairCount, totalValSize);
    endTimer("sort key-values [GPU]", &time_start);

    startTimer(&time_start);
    int *d_uniqueWordOffsets = NULL;
    int *d_wordCounts = NULL;
    int numUniqueWords = 0;
    runReduceByKey(totalPairCount, totalValSize, d_keys, d_vals, &d_uniqueWordOffsets, &d_wordCounts, &numUniqueWords);
    endTimer("reduce by key [GPU]", &time_start);

    startTimer(&time_start);
    int *h_uniqueWordOffsets = (int*)malloc(sizeof(int)*numUniqueWords);
    cudaMemcpy(h_uniqueWordOffsets, d_uniqueWordOffsets, sizeof(int)*numUniqueWords, cudaMemcpyDeviceToHost);

    int *h_wordCounts = (int*)malloc(sizeof(int)*(numUniqueWords));
    cudaMemcpy(h_wordCounts, d_wordCounts, sizeof(int)*(numUniqueWords), cudaMemcpyDeviceToHost);
    endTimer("copy final wordcount to host [GPU->CPU]", &time_start);
    
    int limit = 10;
    printOutput(h_keys, h_uniqueWordOffsets, h_wordCounts, numUniqueWords, limit);
    printf("\n");
    return 0;
}


