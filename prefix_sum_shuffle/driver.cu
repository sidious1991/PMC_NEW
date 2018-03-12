#include <stdio.h>

#include <cuda_runtime.h>

#define MY_CUDA_CHECK(call) {                                    \
    cudaError err = call;                                                    \
    if(cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }}

#define DEFAULTBLOCKSIZE 1024

__inline__ __device__ int lane_id(void){return threadIdx.x%warpSize;}
__inline__ __device__ int warp_id(void){return threadIdx.x/warpSize;}

//Prefix sum among threads within a warp
__inline__ __device__ int warpPrefixSum(int val, int lane_id){
	
	int val_shuffled;
	
	for(int offset = 1; offset < warpSize; offset *=2){	
		
		val_shuffled = __shfl_up(val, offset);
		
		if(lane_id >= offset){
			val += val_shuffled;
		}
	}
	
	return val;
}

//Prefix sum in a block
__inline__ __device__ int blockPrefixSum(int val){
	
	static __shared__ int shared[32]; //Less then 32 warps in a block
	int lane = lane_id(); //Thread id within its warp
	int wid = warp_id(); //Warp id within its block
	int increment;
	
	val = warpPrefixSum(val, lane);
	
	if(lane == (warpSize - 1)){shared[wid] = val;}
	
	__syncthreads();
	
	
	increment = (threadIdx.x < blockDim.x/warpSize)?(shared[lane]):(0); 
	
	if(wid == 0){
		increment = warpPrefixSum(increment, lane);
		shared[lane] = increment;
	}
	
	__syncthreads();
	
	if(wid > 0){
		val += shared[wid - 1];
	}
	
	return val;
}

//Kernel Prefix Sum
__global__ void PrefixSum(int *d_data, int *sums){
	
	//Index of the element to consider
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int val = d_data[index];
	
	val = blockPrefixSum(val);
	
	d_data[index] = val;
	
	if(sums != NULL && threadIdx.x == (blockDim.x - 1)){
		sums[blockIdx.x] = val;
	}
}

//Kernel Apply Increment
__global__ void ApplyIncrement(int *d_data, int *block_sums_scanned){
	
	//Index of the element to consider
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(blockIdx.x > 0){
		d_data[index] += block_sums_scanned[blockIdx.x - 1];
	}
}

void ssb_prefix_sum(int *d_data, int n_elements, int *block_sums) {
	
	int blockSize = DEFAULTBLOCKSIZE;
	
	//How many blocks in the grid
	int gridDim = ((n_elements+blockSize-1)/blockSize);
	
	PrefixSum<<<gridDim, blockSize>>>(d_data, block_sums);
	PrefixSum<<<1,blockSize>>>(block_sums, NULL);
	ApplyIncrement<<<gridDim, blockSize>>>(d_data, block_sums);
}


// This function verifies the shuffle scan result, for the simple
// prefix sum case.
bool CPUverify(int *h_data, int *h_result, int n_elements)
{
    // cpu verify
    for (int i=0; i<n_elements-1; i++)
    {
        h_data[i+1] = h_data[i] + h_data[i+1];
    }

    int diff = 0;

    for (int i=0 ; i<n_elements; i++)
    {
    	//printf("device : %d, host : %d \n",h_result[i], h_data[i]);
        diff += h_data[i]-h_result[i];
    }

    printf("CPU verify result diff (GPUvsCPU) = %d\n", diff);
    bool bTestResult = false;

    if (diff == 0) bTestResult = true;

    return bTestResult;
}

int main(int argc, char **argv) {
    int *h_data, *h_result;
    int *d_data;
    int *block_sums;
    
    int blockSize = DEFAULTBLOCKSIZE;
    int n_elements=65536;
    int n_aligned;
    if(argc>1) {
    	n_elements = atoi(argv[1]);
    }
    n_aligned=((n_elements+blockSize-1)/blockSize)*blockSize;
    int sz = sizeof(int)*n_aligned;

    printf("Starting shfl_scan\n");

    MY_CUDA_CHECK(cudaMallocHost((void **)&h_data, sizeof(int)*n_aligned));
    MY_CUDA_CHECK(cudaMallocHost((void **)&h_result, sizeof(int)*n_elements));

    //initialize data:
    printf("Computing Simple Sum test on %d (%d) elements\n",n_elements, n_aligned);
    printf("---------------------------------------------------\n");

    printf("Initialize test data\n");
    char line[1024];
    for (int i=0; i<n_elements; i++)
    {
        h_data[i] = i;
//        fgets(line,sizeof(line),stdin);
//        sscanf(line,"%d",&h_data[i]);
    }

    for (int i=n_elements; i<n_aligned; i++) {
	h_data[i] = 0;
    }

    printf("Scan summation for %d elements\n", n_elements);

    // initialize a timer
    cudaEvent_t start, stop;
    MY_CUDA_CHECK(cudaEventCreate(&start));
    MY_CUDA_CHECK(cudaEventCreate(&stop));
    float et = 0;
    float inc = 0;

    MY_CUDA_CHECK(cudaMalloc((void **)&block_sums, blockSize*sizeof(int)));
    MY_CUDA_CHECK(cudaMalloc((void **)&d_data, sz));
    MY_CUDA_CHECK(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));
    MY_CUDA_CHECK(cudaEventRecord(start, 0));
    
    ssb_prefix_sum(d_data,n_elements,block_sums);
    
    MY_CUDA_CHECK(cudaEventRecord(stop, 0));
    MY_CUDA_CHECK(cudaEventSynchronize(stop));
    MY_CUDA_CHECK(cudaEventElapsedTime(&inc, start, stop));
    et+=inc;

    MY_CUDA_CHECK(cudaMemcpy(h_result, d_data, n_elements*sizeof(int), cudaMemcpyDeviceToHost));
    printf("Time (ms): %f\n", et);
    printf("%d elements scanned in %f ms -> %f MegaElements/s\n",
             n_elements, et, n_elements/(et/1000.0f)/1000000.0f);

    bool bTestResult = CPUverify(h_data, h_result, n_elements);

    MY_CUDA_CHECK(cudaFreeHost(h_data));
    MY_CUDA_CHECK(cudaFreeHost(h_result));
    MY_CUDA_CHECK(cudaFree(block_sums));
    MY_CUDA_CHECK(cudaFree(d_data));
    MY_CUDA_CHECK(cudaEventDestroy(start));
    MY_CUDA_CHECK(cudaEventDestroy(stop));
    
    return (int)bTestResult;
}
