#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

#define max_err (float)1e-5

#define NSTREAMS 3

#define thread_per_block_xy 32
#define optimum_num_blocks 2 //for multistreaming

#define M 784 //cols firs matrix (divisible by NSTREAMS)
#define N 56 //cols second matrix
#define P 60000 //#of patterns (rows of each matrices)

//#define CARTESIAN_DIM M*N //#of columns of the resulting matrix (cartesian product)
//#define STREAMSIZE M/NSTREAMS

/*---------------------------------------MODALITA' DI PRODOTTO CARTESIANO CHE FA USO DELL'ISTRUZIONE SHUFFLE----------------------------------*/

//__inline__ __device__ int lane_id(void) {
//	return (threadIdx.x + thread_per_block_xy * threadIdx.y) % warpSize;
//}
//
///*Returns 0 if my half warp is the first half warp of the warp*/
//__inline__ __device__ int half_warp(int block_dim) {
//	return (lane_id() >= block_dim);
//}

//__global__ void kernel(float *dev_a, float *dev_b, float *dev_c) {
//
//	int block_x = blockIdx.x;
//	int block_y = blockIdx.y;
//	int thread_x = threadIdx.x;
//	int thread_y = threadIdx.y;
//	int block_dim = blockDim.x; // assumiamo che i blocchi siano quadrati
//
//	//Posizione (x,y) nella matrice B del thread corrente
//	int index_b_x = block_x*block_dim + thread_x;
//	int index_b_y = block_y*block_dim + thread_y;
//
//	//Caricamento del valore di interesse dalla matrice B
//	float val_b = dev_b[index_b_y*N + index_b_x];
//
//	int start_dev_a = block_y*block_dim*M;
//	int step_dev_a = block_dim;
//	int end_dev_a = start_dev_a + M - 1;
//
//	//Posizione corrente dell'indice del for block_a lungo le x (distanza del blocco corrente rispetto al margine)
//	int block_a_x = 0; 
//
//	for (int block_a = start_dev_a; block_a < end_dev_a; block_a += step_dev_a) {
//
//		__shared__ float shared_cartesian[thread_per_block_xy][thread_per_block_xy*thread_per_block_xy];
//
//		//Ciascun thread carica il proprio elemento dalla matrice A
//		float val_a = dev_a[block_a + thread_y*M + thread_x];
//		//A questo punto ciascun thread in un determinato warp ha caricato il proprio val_a in un registro...
//		int my_half_warp = half_warp(block_dim);
//
//		#pragma unroll (thread_per_block_xy)
//		for (int src_lane = my_half_warp*block_dim; src_lane < block_dim*(1 + my_half_warp); src_lane++) {
//			
//			//Leggo i registri degli altri threads del mio stesso half warp
//			float src_a = __shfl(val_a, src_lane);
//			//Calcolo del prodotto dell'elemento j di A (lungo le x) e dell'elemento k di B (lungo le x)
//			float current_prod = src_a*val_b;
//
//			//posizione lungo le y, all'interno del suo blocco, del valore considerato in A (o equivalentemente in B)
//			int patt_index = thread_y;
//			//posizione lungo le x , all'interno del suo blocco, del valore considerato nella matrice A (ossia quello ottenuto dallo shuffle --> src_a)
//			int j = src_lane%block_dim; 
//			//posizione lungo le x, all'interno del suo blocco, del valore considerato nella matrice B (ossia val_b)
//			int k = thread_x;
//
//			//Salvataggio nella shared del prodotto cartesiano corrente
//			shared_cartesian[patt_index][j*block_dim + k] = current_prod;
//		}
//
//		__syncthreads();
//
//		float reduced_value = 0.0f;
//		//Adesso ogni thread del blocco è chiamato a ridurre in modo indipendente la sua coppia (j,k) scorrendo le righe (ossia i patterns)
//		#pragma unroll (thread_per_block_xy)
//		for (int patt = 0; patt < block_dim; patt++) {
//			reduced_value += shared_cartesian[patt][thread_y*block_dim + thread_x];
//			//thread_y corrisponde all'indice j della matrice shared e thread_x corrisponde all'indice k della matrice shared
//		}
//
//		//Here atomicAdd...
//		int j_index_dest = (thread_y + block_a_x)*N;
//		int k_index_dest = thread_x + block_x*block_dim;
//
//		atomicAdd(&dev_c[j_index_dest + k_index_dest], reduced_value);
//
//		block_a_x += block_dim;
//	}
//}
//
//
//void cartesian_product(float *dev_a, float *dev_b, float *dev_c) {
//
//	dim3 block;
//	dim3 grid;
//
//	block.x = thread_per_block_xy;
//	block.y = thread_per_block_xy;
//
//	//Il prodotto è A X B = C e la griglia si dispone sulla matrice B (P righe e N colonne).
//	grid.x = (N + thread_per_block_xy - 1) / (thread_per_block_xy);
//	grid.y = (P + thread_per_block_xy - 1) / (thread_per_block_xy);
//
//	kernel << <grid, block >> > (dev_a, dev_b, dev_c);
//}

/*---------------------------------------MODALITA' DI PRODOTTO CARTESIANO CON GRIGLIA POSIZIONATA SULL'USCITA----------------------------------*/

//__device__ void cartesian_dev(float *dev_a, float *dev_b, float *dev_c, int offset) {
//
//	int c_index_j = blockIdx.y*blockDim.y + threadIdx.y;
//	int c_index_k = blockIdx.x*blockDim.x + threadIdx.x;
//
//	if ((c_index_j + offset) < M && c_index_k < N) {
//		
//		float current_prod = 0.0f;
//		
//		#pragma unroll(2)
//		for (int p = 0; p < P; p++) {
//			current_prod += dev_a[p*M + c_index_j + offset] * dev_b[p*N + c_index_k];
//		}
//		//Non vi è bisogno di atomicAdd poichè in quest'altra implementazione ogni thread gestisce una e una sola posizione (j,k) dell'output
//		dev_c[(c_index_j+offset)*N + c_index_k] = current_prod;
//	}
//}
//
//__global__ void kernel(float *dev_a, float *dev_b, float *dev_c, int offset) {
//	
//	int patt_per_step = gridDim.y*blockDim.y;
//
//	for (int j = 0; j < STREAMSIZE; j+=patt_per_step) {
//		cartesian_dev(dev_a, dev_b, dev_c, offset+j);
//	}
//}
//
//void cartesian_product(float *dev_a, float *dev_b, float *dev_c) {
//
//	dim3 block;
//	dim3 grid;
//
//	block.x = thread_per_block_xy;
//	block.y = thread_per_block_xy;
//
//	grid.x = (N + thread_per_block_xy - 1) / (thread_per_block_xy);
//	grid.y = MAX(optimum_num_blocks/grid.x, 1);
//
//	cudaStream_t streams[NSTREAMS];
//	int offset;
//
//	for (int s = 0; s < NSTREAMS; s++) {
//
//		cudaStreamCreate(&streams[s]);
//		offset = s*STREAMSIZE;
//
//		kernel << <grid, block, 0, streams[s] >> > (dev_a, dev_b, dev_c, offset);
//	}
//}

/*---------------------------------------MODALITA' DI PRODOTTO CARTESIANO CON PRODOTTO MATRICE-MATRICE----------------------------------*/

__device__ void MMMulDevPartialCartesian(float *dev_a, float *dev_b, float *dev_c, int current_rows, int patterns) {

	int tx = threadIdx.x, ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int block_dim = blockDim.x; // assumiamo che i blocchi siano quadrati
	int dest_x = block_x*block_dim + tx;
	int dest_y = block_y*block_dim + ty;

	int dev_a_x = block_y*block_dim;
	int dev_b_x = block_x*block_dim;

	int end_dev_a = dev_a_x + M*(patterns - 1);

	int step_dev_a = M*block_dim;
	int step_dev_b = N*block_dim;

	float partial = 0.0f;
	int block_r_border = 0; // contatore che indica in che iterazione dei blocchi ci troviamo
	int current_inc;
	int min;

	for (int dev_a_id = dev_a_x, dev_b_id = dev_b_x; dev_a_id <= end_dev_a; dev_a_id += step_dev_a, dev_b_id += step_dev_b) {

		block_r_border += block_dim;

		__shared__ float shared_dev_a[thread_per_block_xy][thread_per_block_xy];
		__shared__ float shared_dev_b[thread_per_block_xy][thread_per_block_xy];

		int t_index_dev_a = dev_a_id + tx + ty*M;
		int t_index_dev_b = dev_b_id + tx + ty*N;

		//Salviamo la sottomatrice trasposta nella shared memory della matrice dev_a (osservare che in tal modo evitiamo conflitti di banco):
		shared_dev_a[tx][ty] = (t_index_dev_a < patterns*M) ? (dev_a[t_index_dev_a]) : (0.0f);
		shared_dev_b[ty][tx] = (t_index_dev_b < patterns*N) ? (dev_b[t_index_dev_b]) : (0.0f);

		__syncthreads();

		current_inc = patterns - (block_r_border - block_dim);
		min = MIN(current_inc, block_dim);

		#pragma unroll(2)
		for (int k = 0; k < min; k++) {
			partial += shared_dev_a[ty][k] * shared_dev_b[k][tx];
		}

		__syncthreads();
	}

	if (dest_x < N && dest_y < current_rows) {
		atomicAdd(&dev_c[dest_y*N + dest_x], partial);
	}
}

__global__ void kernel(float *dev_a, float *dev_b, float *dev_c, int patterns) {

	int current_rows;
	int remaining_rows;
	int pos_block_y = blockIdx.y*blockDim.x; //Posizione del blocco corrente rispetto alla griglia lungo le y
											 //Assumiamo che i blocchi siano quadrati (blockDim.x = blockDim.y)		
	int rows_per_step = gridDim.y*blockDim.y; //Righe della matrice delle derivate dei pesi da considerare (al massimo) ogni step

	for (int y = 0; y < M; y += rows_per_step) {

		remaining_rows = M - y;
		current_rows = MIN(remaining_rows, rows_per_step);

		if (pos_block_y >= current_rows) { return; }
		MMMulDevPartialCartesian(dev_a + y, dev_b, dev_c + y*N, current_rows, patterns);
	}
}

void cartesian_product(float *dev_a, float *dev_b, float *dev_c) {

	dim3 block;
	dim3 grid;

	block.x = thread_per_block_xy;
	block.y = thread_per_block_xy;

	grid.x = (N + thread_per_block_xy - 1) / (thread_per_block_xy);
	grid.y = MAX(optimum_num_blocks/grid.x, 1);

	int stream_size = P / NSTREAMS;
	//Se il numero dei pattern non fosse divisibile per NSTREAMS avremmo streams_remainder > 0
	int stream_remainder = P % NSTREAMS;
	
	cudaStream_t streams[NSTREAMS];
	int offset;

	for (int s = 0; s < NSTREAMS; s++) {
		
		cudaStreamCreate(&streams[s]);
		offset = s*stream_size;
		if (s == (NSTREAMS - 1)) { stream_size += stream_remainder; }

		kernel << <grid, block, 0, streams[s] >> > (dev_a + offset*M, dev_b + offset*N, dev_c, stream_size);
	}
}

/*-----------------------------------------------------------END DEVICE---------------------------------------------------------*/

void printMat(float *mat, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		printf("Row %d: {", i);
		for (int j = 0; j < cols; j++) {
			printf("%f,	 ", mat[i*cols + j]);
		}
		printf("}\n\n");
	}
}

void cartesian_product_host(float *a, float *b, float *c) {

	for (int p = 0; p < P; p++) {
		for (int j = 0; j < M; j++) {
			for (int k = 0; k < N; k++) {
				c[j*N + k] += a[p*M + j] * b[p*N + k];
			}
		}
	}
}

int mat_equals(float *a, float *b) {

	for (int j = 0; j < M; j++) {
		for (int k = 0; k < N; k++) {
			if (fabs(a[j*N + k]-b[j*N + k])>max_err) { 
				printf("error : %f\n", fabs(a[j*N + k] - b[j*N + k]));
				return 0; 
			}
		}
	}
	return 1;
}

void compare_mats(float *a, float *b, int rows, int cols) {

	for (int j = 0; j < M; j++) {
		for (int k = 0; k < N; k++) {
			printf("(%d,%d): %f,%f\n", j, k, a[j*N + k], b[j*N + k]);
		}
	}
}


int main()
{

	float *a, *b, *c, *c_host, *dev_a, *dev_b, *dev_c;
	a = (float*)malloc(P*M * sizeof(float));
	b = (float*)malloc(P*N * sizeof(float));
	c = (float*)malloc(M*N * sizeof(float));
	c_host = (float*)calloc(M*N, sizeof(float));
	cudaMalloc(&dev_a, P*M * sizeof(float));
	cudaMalloc(&dev_b, P*N * sizeof(float));
	cudaMalloc(&dev_c, M*N * sizeof(float));

	for (int i = 0; i < P*M; i++) {
		a[i] = (float)rand() / (float)RAND_MAX;
	}

	for (int i = 0; i < P*N; i++) {
		b[i] = (float)rand() / (float)RAND_MAX;
	}

	cudaMemcpy(dev_a, a, P*M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, P*N * sizeof(float), cudaMemcpyHostToDevice);

	cartesian_product(dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, M*N * sizeof(float), cudaMemcpyDeviceToHost);


	cartesian_product_host(a, b, c_host);
	
	/*
	int eq = mat_equals(c, c_host);
	printf("Are equals: %d\n", eq);
	*/
	compare_mats(c_host, c, M, N);
	
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	free(a);
	free(b);
	free(c);
	free(c_host);

    return 0;
}

