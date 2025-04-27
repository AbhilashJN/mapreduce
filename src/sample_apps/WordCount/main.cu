#include "MarsInc.h"
#include "global.h"
#include <ctype.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define __OUTPUT__

// Toggle between original and async version
#define USE_ASYNC 1

void validate(char* h_filebuf, Spec_t* spec, int num)
{
    char* key = (char*)spec->outputKeys;
    char* val = (char*)spec->outputVals;
    int4* offsetSizes = (int4*)spec->outputOffsetSizes;
    int2* range = (int2*)spec->outputKeyListRange;

    printf("# of words: %d\n", spec->outputDiffKeyCount);
    if (num > spec->outputDiffKeyCount) num = spec->outputDiffKeyCount;
    for (int i = 0; i < num; i++)
    {
        int keyOffset = offsetSizes[range[i].x].x;
        int valOffset = offsetSizes[range[i].x].z;
        char* word = key + keyOffset;
        int wordsize = *(int*)(val + valOffset);
        printf("%s - size: %d - count: %d\n", word, wordsize, range[i].y - range[i].x);
    }
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Usage: %s datafile\n", argv[0]);
        exit(-1);
    }

#if USE_ASYNC
    printf("Running in ASYNC mode (cudaMallocAsync + cudaMemcpyAsync + pinned memory)\n");
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);
#else
    printf("Running in ORIGINAL mode (cudaMalloc + cudaMemcpy + malloc)\n");
#endif

    Spec_t* spec = GetDefaultSpec();
    spec->workflow = MAP_GROUP;
#ifdef __OUTPUT__
    spec->outputToHost = 1;
#endif

    FILE* fp = fopen(argv[1], "r");
    fseek(fp, 0, SEEK_END);
    int fileSize = ftell(fp) + 1;
    rewind(fp);

    // Host buffer allocation
    char* h_filebuf = NULL;
#if USE_ASYNC
    cudaMallocHost((void**)&h_filebuf, fileSize);  // Pinned memory for async
#else
    h_filebuf = (char*)malloc(fileSize);           // Normal malloc for blocking
#endif
    fread(h_filebuf, fileSize, 1, fp);
    fclose(fp);

    // Device buffer allocation
    char* d_filebuf = NULL;
#if USE_ASYNC
    cudaMallocAsync((void**)&d_filebuf, fileSize, stream);
#else
    cudaMalloc((void**)&d_filebuf, fileSize);
#endif

    WC_KEY_T key;
    key.file = d_filebuf;

    for (int i = 0; i < fileSize; i++)
        h_filebuf[i] = toupper(h_filebuf[i]);

    WC_VAL_T val;
    int offset = 0;
    char* p = h_filebuf;
    char* start = h_filebuf;

    while (1)
    {
        int blockSize = 2048;
        if (offset + blockSize > fileSize) blockSize = fileSize - offset;
        p += blockSize;
        for (; *p >= 'A' && *p <= 'Z'; p++);

        if (*p != '\0')
        {
            *p = '\0';
            ++p;
            blockSize = (int)(p - start);
            val.line_offset = offset;
            val.line_size = blockSize;
            AddMapInputRecord(spec, &key, &val, sizeof(WC_KEY_T), sizeof(WC_VAL_T));
            offset += blockSize;
            start = p;
        }
        else
        {
            *p = '\0';
            blockSize = (int)(fileSize - offset);
            val.line_offset = offset;
            val.line_size = blockSize;
            AddMapInputRecord(spec, &key, &val, sizeof(WC_KEY_T), sizeof(WC_VAL_T));
            break;
        }
    }

    //----------------------------------------------
    // GPU Memory Copy
    //----------------------------------------------
#if USE_ASYNC
    cudaMemcpyAsync(d_filebuf, h_filebuf, fileSize, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
#else
    cudaMemcpy(d_filebuf, h_filebuf, fileSize, cudaMemcpyHostToDevice);
#endif

    //----------------------------------------------
    // MapReduce work
    //----------------------------------------------
    MapReduce(spec);

#ifdef __OUTPUT__
#if USE_ASYNC
    cudaMemcpyAsync(h_filebuf, d_filebuf, fileSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
#else
    cudaMemcpy(h_filebuf, d_filebuf, fileSize, cudaMemcpyDeviceToHost);
#endif
    validate(h_filebuf, spec, 10);
#endif

    //----------------------------------------------
    // Finish
    //----------------------------------------------
    FinishMapReduce(spec);

#if USE_ASYNC
    cudaFreeAsync(d_filebuf, stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_filebuf);  // Free pinned memory
#else
    cudaFree(d_filebuf);
    free(h_filebuf);          // Free regular memory
#endif

    return 0;
}
