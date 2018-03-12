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
#define ELEMPERBLOCK DEFAULTBLOCKSIZE*2 //1 thread per 2 elements

#define NUMBANKS 32
#define LOGNUMBANKS 5 //NumBanks = 32
#define CONFLICT_FREE_OFFSET(n) ((n)>>(LOGNUMBANKS))

//Kernel (per block)
__global__ void scan(int *d_data_input, int *d_data_output, int n_block){
	
	extern __shared__ int temp[];
	
	int tid = threadIdx.x;
	int offset = 1;
	
	int ai = tid;
	int bi = tid + (n_block/2);
	
	int offset_a = CONFLICT_FREE_OFFSET(ai);
	int offset_b = CONFLICT_FREE_OFFSET(bi);
		
	temp[ai + offset_a] =  d_data_input[ai + blockIdx.x*blockDim.x];
	temp[bi + offset_b] =  d_data_input[bi + blockIdx.x*blockDim.x];
			
	for (int d = n_block>>1; d > 0; d >>= 1){ 
		__syncthreads();
	   if (tid < d){
		   int ai = offset*(2*tid+1)-1;
		   int bi = offset*(2*tid+2)-1;
		   ai += CONFLICT_FREE_OFFSET(ai);
		   bi += CONFLICT_FREE_OFFSET(bi);
		   temp[bi] += temp[ai];
	   }
	   offset *= 2;
	}
	
	if (tid==0) { temp[n_block - 1 + CONFLICT_FREE_OFFSET(n_block - 1)] = 0;}
	
	for (int d = 1; d < n_block; d *= 2){
		offset >>= 1;
		__syncthreads();
		if (tid < d){
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t; 
		}
	}
	__syncthreads();
	
	d_data_output[ai + blockIdx.x*blockDim.x] = temp[ai + offset_a]; // write results to device memory
	d_data_output[bi + blockIdx.x*blockDim.x] = temp[bi + offset_b];
	
}

void ssb_prefix_sum(int *d_data_input, int *d_data_output, int n_elements) {
	
	int elemperblock = ELEMPERBLOCK;
	int num_padding_shared = (elemperblock/NUMBANKS)-1;
	
	//How many blocks in the grid (1024 block threads for 2048 elements of d_data_input)
	int gridDim = ((n_elements+elemperblock-1)/elemperblock);	
	
	scan<<<gridDim, DEFAULTBLOCKSIZE, (elemperblock+num_padding_shared)*sizeof(int)>>>(d_data_input, d_data_output, elemperblock);
}


// This function verifies the scan result, for the simple
// prefix sum case.
bool CPUverify(int *h_data, int *h_result, int n_elements)
{
	int out[n_elements];//temporary
	out[0] = 0;
    // cpu verify
    for (int i=0; i<n_elements-1; i++)
    {
        out[i+1] = h_data[i] + out[i];
    }

    int diff = 0;

    for (int i=0 ; i<n_elements; i++)
    {
    	//printf("%d,%d\n",h_result[i],out[i]);
        diff += out[i]-h_result[i];
    }

    printf("CPU verify result diff (GPUvsCPU) = %d\n", diff);
    bool bTestResult = false;

    if (diff == 0) bTestResult = true;

    return bTestResult;
}

int main(int argc, char **argv) {
	
    int *h_data, *h_result;
    int *d_data_input, *d_data_output;//Input and output(result) for device
    int elemperblock = ELEMPERBLOCK;
    int n_elements=65536;
    int n_aligned;
    if(argc>1) {
    	n_elements = atoi(argv[1]);
    }
    n_aligned=((n_elements+elemperblock-1)/elemperblock)*elemperblock;
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

    MY_CUDA_CHECK(cudaMalloc((void **)&d_data_input, sz));
    MY_CUDA_CHECK(cudaMalloc((void **)&d_data_output, sz));
    MY_CUDA_CHECK(cudaMemcpy(d_data_input, h_data, sz, cudaMemcpyHostToDevice));
    
    MY_CUDA_CHECK(cudaEventRecord(start, 0));
  
    ssb_prefix_sum(d_data_input, d_data_output, n_elements);
  
    MY_CUDA_CHECK(cudaEventRecord(stop, 0));
    MY_CUDA_CHECK(cudaEventSynchronize(stop));
    MY_CUDA_CHECK(cudaEventElapsedTime(&inc, start, stop));
    et+=inc;

    MY_CUDA_CHECK(cudaMemcpy(h_result, d_data_output, n_elements*sizeof(int), cudaMemcpyDeviceToHost));
    printf("Time (ms): %f\n", et);
    printf("%d elements scanned in %f ms -> %f MegaElements/s\n",
             n_elements, et, n_elements/(et/1000.0f)/1000000.0f);

    bool bTestResult = CPUverify(h_data, h_result, n_elements);

    MY_CUDA_CHECK(cudaFreeHost(h_data));
    MY_CUDA_CHECK(cudaFreeHost(h_result));
    MY_CUDA_CHECK(cudaFree(d_data_input));
    MY_CUDA_CHECK(cudaFree(d_data_output));
    MY_CUDA_CHECK(cudaEventDestroy(start));
    MY_CUDA_CHECK(cudaEventDestroy(stop));
    
    return (int)bTestResult;
}