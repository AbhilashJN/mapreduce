#include "device_utils.cuh"

__device__ void d_printBlock(char *filebuf, size_t start, size_t len){
    char *temp = (char *)malloc(len+1);
    temp[len]='\0';
    for(size_t i=0;i<len;i++){
        temp[i] = filebuf[start+i];
    }
    printf("%s\n",temp);
    free(temp);
}


void doPrefixSum(int *d_ptr,size_t numItems){
    thrust::device_ptr<int> td_paircounts = thrust::device_pointer_cast(d_ptr);
    thrust::exclusive_scan(td_paircounts, td_paircounts+numItems, td_paircounts);

    int *prefixSums = (int*)malloc(numItems * sizeof(int));
    cudaMemcpy(prefixSums, d_ptr,numItems * sizeof(int),  cudaMemcpyDeviceToHost);

    free(prefixSums);
}

