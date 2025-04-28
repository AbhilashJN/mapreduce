#include "MarsInc.h"
#include "global.h"
#include <ctype.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <algorithm>
#include <string.h>

#define __OUTPUT__

// Toggle between original and async version
#define USE_ASYNC 1

// Chunking thresholds
const size_t THRESHOLD_SIZE = 128 * 1024 * 1024; // 128MB
const size_t CHUNK_SIZE = 512 * 1024 * 1024;     // 512MB

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

//--------------------------------------------------------
// Copy a chunk's output and add to globalSpec input
//--------------------------------------------------------
void accumulateChunkOutputToGlobal(Spec_t* chunkSpec, Spec_t* globalSpec)
{
    char* keys = (char*)chunkSpec->outputKeys;
    char* vals = (char*)chunkSpec->outputVals;
    int4* offsetSizes = (int4*)chunkSpec->outputOffsetSizes;
    int2* keyListRange = (int2*)chunkSpec->outputKeyListRange;
    int wordCount = chunkSpec->outputDiffKeyCount;

    for (int i = 0; i < wordCount; i++)
    {
        int keyOffset = offsetSizes[keyListRange[i].x].x;
        int valOffset = offsetSizes[keyListRange[i].x].z;
        char* word = keys + keyOffset;
        int count = keyListRange[i].y - keyListRange[i].x;

        // Add to the new Spec_t
        AddMapInputRecord(globalSpec, word, &count, strlen(word)+1, sizeof(int));
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

    FILE* fp = fopen(argv[1], "r");
    fseek(fp, 0, SEEK_END);
    size_t fileSize = ftell(fp) + 1;
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

    // Uppercase all characters
    for (size_t i = 0; i < fileSize; i++)
        h_filebuf[i] = toupper(h_filebuf[i]);

    // Global collector Spec for all chunk outputs
    Spec_t* globalSpec = GetDefaultSpec();
    globalSpec->workflow = MAP_GROUP;
#ifdef __OUTPUT__
    globalSpec->outputToHost = 1;
#endif

    //----------------------------------------------
    // SMALL FILE PATH
    //----------------------------------------------
    if (fileSize <= THRESHOLD_SIZE)
    {
        Spec_t* spec = GetDefaultSpec();
        spec->workflow = MAP_GROUP;
#ifdef __OUTPUT__
        spec->outputToHost = 1;
#endif

        // Device memory
        char* d_filebuf = NULL;
#if USE_ASYNC
        cudaMallocAsync((void**)&d_filebuf, fileSize, stream);
        cudaMemcpyAsync(d_filebuf, h_filebuf, fileSize, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
#else
        cudaMalloc((void**)&d_filebuf, fileSize);
        cudaMemcpy(d_filebuf, h_filebuf, fileSize, cudaMemcpyHostToDevice);
#endif

        // Prepare input records
        WC_KEY_T key;
        key.file = d_filebuf;
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
        MapReduce(spec);

#ifdef __OUTPUT__
        validate(h_filebuf, spec, 10);
#endif

        FinishMapReduce(spec);

#if USE_ASYNC
        cudaFreeAsync(d_filebuf, stream);
#else
        cudaFree(d_filebuf);
#endif
    }
    //----------------------------------------------
    // LARGE FILE PATH: Chunked
    //----------------------------------------------
    else
    {
        printf("Large file detected (size: %zu bytes), using chunked processing\n", fileSize);

        // Allocate device buffer
        char* d_filebuf = NULL;
#if USE_ASYNC
        cudaMallocAsync((void**)&d_filebuf, CHUNK_SIZE, stream);
#else
        cudaMalloc((void**)&d_filebuf, CHUNK_SIZE);
#endif

        size_t offset = 0;
        while (offset < fileSize)
        {
            size_t currentChunkSize = std::min(CHUNK_SIZE, fileSize - offset);

            // Allocate chunk spec
            Spec_t* chunkSpec = GetDefaultSpec();
            chunkSpec->workflow = MAP_GROUP;
            chunkSpec->outputToHost = 1;

            // Prepare chunk input records
            WC_KEY_T key;
            key.file = d_filebuf;
            WC_VAL_T val;
            char* p = h_filebuf + offset;
            char* start = p;
            size_t chunkEnd = offset + currentChunkSize;

            int localOffset = offset;
            while (p < h_filebuf + chunkEnd)
            {
                int blockSize = 2048;
                if (p + blockSize > h_filebuf + chunkEnd)
                    blockSize = (h_filebuf + chunkEnd) - p;
                p += blockSize;
                for (; *p >= 'A' && *p <= 'Z'; p++);

                if (*p != '\0' && p < h_filebuf + chunkEnd)
                {
                    *p = '\0';
                    ++p;
                    blockSize = (int)(p - start);
                    val.line_offset = start - h_filebuf;
                    val.line_size = blockSize;
                    AddMapInputRecord(chunkSpec, &key, &val, sizeof(WC_KEY_T), sizeof(WC_VAL_T));
                    start = p;
                }
                else
                {
                    *p = '\0';
                    blockSize = (h_filebuf + chunkEnd) - start;
                    val.line_offset = start - h_filebuf;
                    val.line_size = blockSize;
                    AddMapInputRecord(chunkSpec, &key, &val, sizeof(WC_KEY_T), sizeof(WC_VAL_T));
                    break;
                }
            }

            // Copy and MapReduce the chunk
#if USE_ASYNC
            cudaMemcpyAsync(d_filebuf, h_filebuf + offset, currentChunkSize, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
#else
            cudaMemcpy(d_filebuf, h_filebuf + offset, currentChunkSize, cudaMemcpyHostToDevice);
#endif
            MapReduce(chunkSpec);

            // Accumulate output into globalSpec
            accumulateChunkOutputToGlobal(chunkSpec, globalSpec);

            FinishMapReduce(chunkSpec);

            offset += currentChunkSize;
        }

#if USE_ASYNC
        cudaFreeAsync(d_filebuf, stream);
#else
        cudaFree(d_filebuf);
#endif

        //----------------------------------------------
        // Final global MapReduce pass
        //----------------------------------------------
        printf("Performing final aggregation MapReduce across all chunk outputs...\n");
        MapReduce(globalSpec);

#ifdef __OUTPUT__
        validate((char*)globalSpec->outputKeys, globalSpec, 10);
#endif

        FinishMapReduce(globalSpec);
    }

    //----------------------------------------------
    // Cleanup
    //----------------------------------------------
#if USE_ASYNC
    cudaStreamDestroy(stream);
    cudaFreeHost(h_filebuf);
#else
    free(h_filebuf);
#endif

    return 0;
}
