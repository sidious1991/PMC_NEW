#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DATA float
#define BOOL int
#define MAX_ERR 1e-5

//Grid features

#define OPTIMUM_BLOCK_NUM 12 
#define BLOCK_SIDE	16 

#define OPTIMUM_BLOCK_NUM_FIRST_LAYER 2
#define BLOCK_SIDE_FIRST_LAYER 32

//Network features

#define NEURO_INPUT 784 //#neurons of input layer
#define NEURO_H_0	56	//#neurons of first hidden layer
#define NEURO_H_1	28	//#neurons of second hidden layer
#define NEURO_OUTPUT 10 //#neurons of output layer
#define TOTAL_PATT	60000 //#total patterns
#define NUM_HIDDEN 2 //#hidden layers
#define TOTAL_LAYER 4 //#of layers

//struct features
#define MATRIX_NUMBER_STRUCT 3 //#matrix to copy to Device(in struct)
#define GLOBAL_H_SIZE TOTAL_PATT * (NEURO_H_0 + NEURO_H_1 +NEURO_OUTPUT)
#define DEVICE_GLOBAL_H_SIZE TOTAL_PATT * (NEURO_INPUT + NEURO_H_0 + NEURO_H_1 + NEURO_OUTPUT)
#define GLOBAL_W_SIZE (NEURO_INPUT*NEURO_H_0) + (NEURO_H_0*NEURO_H_1) + (NEURO_H_1*NEURO_OUTPUT)
#define GLOBAL_BIAS_SIZE NEURO_H_0 + NEURO_H_1 + NEURO_OUTPUT
#define HOST_TO_DEVICE_COPY_SIZE GLOBAL_BIAS_SIZE + GLOBAL_W_SIZE

//Streams Settings
#define NSTREAMS 3
#define STREAMSIZE TOTAL_PATT/NSTREAMS

/*Struct Grid Settings*/

typedef struct grid_settings {
	unsigned int grid[3];
	unsigned int block[3];
}grid_settings;

grid_settings gs = { { OPTIMUM_BLOCK_NUM_FIRST_LAYER, OPTIMUM_BLOCK_NUM, OPTIMUM_BLOCK_NUM },{ BLOCK_SIDE_FIRST_LAYER,BLOCK_SIDE,BLOCK_SIDE } };

/*Struct One Copy HostToDev*/
typedef struct host_to_dev_mem{
	DATA WeightH2H[GLOBAL_W_SIZE];
	DATA BiasH2H[GLOBAL_BIAS_SIZE];	
	DATA H2H[DEVICE_GLOBAL_H_SIZE];	
	int matrix_WB_index[MATRIX_NUMBER_STRUCT-1][TOTAL_LAYER-1];//INDEX for padding in Weight & Bias 
	int matrix_H2H_index[MATRIX_NUMBER_STRUCT-2][TOTAL_LAYER];//INDEX for padding in H2H
} host_to_dev_mem;

typedef struct dev_host_to_dev_mem{
	DATA WeightH2H[GLOBAL_W_SIZE];
	DATA BiasH2H[GLOBAL_BIAS_SIZE];	
	DATA H2H[DEVICE_GLOBAL_H_SIZE];	
} dev_host_to_dev_mem;

/*UTILITIES*/

static void HandleCuda(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_CUDA( err ) (HandleCuda( err, __FILE__, __LINE__ ))

void startTimer(cudaEvent_t *start, cudaEvent_t *stop) {
	HANDLE_CUDA(cudaEventCreate(start));
	HANDLE_CUDA(cudaEventCreate(stop));
	HANDLE_CUDA(cudaEventRecord(*start, 0));
}

void stopAndPrint(cudaEvent_t *start, cudaEvent_t *stop) {
	HANDLE_CUDA(cudaEventRecord(*stop, 0));
	HANDLE_CUDA(cudaEventSynchronize(*stop));
	float time = 0.0f;
	HANDLE_CUDA(cudaEventElapsedTime(&time, *start, *stop));
	printf("Elapsed Time: %f milliseconds\n", time);
	HANDLE_CUDA(cudaEventDestroy(*start));
	HANDLE_CUDA(cudaEventDestroy(*stop));
}

/*DEVICE*/

__global__ void MMMulDevPartial(DATA *, DATA *, DATA *, DATA *, unsigned int, unsigned int, unsigned int);
void MMMulDev(DATA *, DATA *, DATA *, DATA *, unsigned int, unsigned int, unsigned int, dim3, dim3, cudaStream_t);

/*HOST*/
//void feedforward(DATA *, DATA **, DATA **, DATA **, DATA **, DATA **, int *, int);
void feedforward(DATA *, struct host_to_dev_mem*, struct dev_host_to_dev_mem*, int *, int);

void HOST_feedforward(DATA *, DATA **, DATA **, DATA **, int *);
void printMat(DATA *, int, int);
void MMMulHost(DATA *, DATA *, DATA *, DATA *, int, int, int);
BOOL matsAreEquals(DATA *, DATA *, int, int);

/*HOST ALLOCATION AND INITIALIZATION*/
void HOST_init_struct(struct host_to_dev_mem* , int* );
void HOST_alloc_init(DATA**, int*);

/*DEVICE ALLOCATION*/


/*HOST DEALLOCATION FUNCTIONS*/


/*DEVICE DEALLOCATION FUNCTIONS*/

void CUDA_dealloc(DATA**, DATA**, DATA**);

/*----------------------------------------------------------------------MAIN---------------------------------------------------------------------------*/

int main(void) {

	DATA *INPUT_MAT;
	int *nupl = (int*)malloc(TOTAL_LAYER * sizeof(int));
	struct host_to_dev_mem *htdm = (struct host_to_dev_mem*)malloc(sizeof(struct host_to_dev_mem));
	struct dev_host_to_dev_mem *dev_htdm;

	/*questa parte bisogner� renderla dinamica. In seguito bisogner� accedere ai files.*/
	nupl[0] = NEURO_INPUT;
	nupl[1] = NEURO_H_0;
	nupl[2] = NEURO_H_1;
	nupl[TOTAL_LAYER - 1] = NEURO_OUTPUT;

	/*host memory allocation and initialization*/
	//pinned
	HANDLE_CUDA(cudaHostAlloc(&INPUT_MAT, nupl[0] * TOTAL_PATT * sizeof(DATA), 0));
	//DATA r;
	for (int i = 0; i < TOTAL_PATT; i++) {
		for (int j = 0; j < NEURO_INPUT; j++) {
			//r= rand() / (DATA)RAND_MAX;
			INPUT_MAT[i*NEURO_INPUT + j] = rand() / (DATA)RAND_MAX;
			//htdm->H2H[i*NEURO_INPUT+ j] = r;
		}
	}
	
	HOST_init_struct(htdm,nupl);
	HANDLE_CUDA(cudaMalloc((void **)&dev_htdm, sizeof(struct dev_host_to_dev_mem)));
	
	/*-----------------------------------FEEDFORWARD-------------------------------------------*/

	cudaEvent_t start, stop;

	startTimer(&start, &stop);
//	feedforward(INPUT_MAT, W_MAT, BIAS_MAT, DEV_H2H_MAT, DEV_W_MAT, DEV_BIAS_MAT, nupl, TOTAL_LAYER);
	feedforward(INPUT_MAT, htdm, dev_htdm, nupl, TOTAL_LAYER);
	stopAndPrint(&start, &stop);
	cudaDeviceSynchronize(); 

	/*-----------------------------END---FEEDFORWARD-------------------------------------------*/

	//Host dealloc
	free(nupl);
	free(htdm);
	cudaFree(dev_htdm);
	cudaFreeHost(INPUT_MAT); 
	//Free temporaray data

	return 0;
}


/*---------------------------------------------------------------------KERNEL--------------------------------------------------------------------------*/

/*DEVICE*/

/* h2h � il puntatore alla porzione dell'h2h globale da considerare in questa fase
(ad ogni passo il kernel che invoca questo device incrementa il puntatore h2h
in modo proporzionale al patt_per_step (e similmente h2h_dest) (vedi sotto))*/

__global__ void MMMulDevPartial(DATA *h2h, DATA *w, DATA *biases, DATA * h2h_dest, unsigned int row_w, unsigned int col_w, unsigned int num_pattern) {
	const int pos_block_y = blockIdx.y*blockDim.x; //Posizione del blocco corrente rispetto alla griglia lungo le y
	if (pos_block_y >= num_pattern) { return; }

	int tx = threadIdx.x, ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	const int block_dim = blockDim.x; // assumiamo che i blocchi siano quadrati
	int dest_x = block_x*block_dim + tx;
	int dest_y = block_y*block_dim + ty;

	int w_x = block_x*block_dim; // start block in w
	int h2h_y = block_y*block_dim*row_w; // start block in h2h

	int end_h2h = h2h_y + row_w - 1; // last block position in h2h

	int step_w = block_dim*col_w;
	int step_h2h = block_dim;
	int min;

	DATA partial = 0.0f;
	int block_r_border = 0; // contatore che indica in che iterazione dei blocchi ci troviamo
	int current_inc;

	for (int wid = w_x, h2h_id = h2h_y; h2h_id <= end_h2h; wid += step_w, h2h_id += step_h2h) {

		block_r_border += block_dim;

		//__shared__ DATA shared_w[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER+1]; Non possiamo ancora giustificare il miglioramento nei tempi.
		__shared__ DATA shared_w[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER];
		__shared__ DATA shared_h2h[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER];

		int t_index_w = wid + tx + ty*col_w;
		int t_index_h2h = h2h_id + tx + ty*row_w;

		//Attenzione alla divergenza dei threads (vedi CCC pag.137)
		shared_h2h[ty][tx] = (t_index_h2h < num_pattern*row_w) ? (h2h[t_index_h2h]) : (0.0f);
		shared_w[ty][tx] = (t_index_w < col_w*row_w) ? (w[t_index_w]) : (0.0f);

		__syncthreads();

		current_inc = row_w - (block_r_border - block_dim);

		min = (current_inc < block_dim) ? (current_inc) : (block_dim);

		for (int k = 0; k < min; k++) {
			partial += shared_h2h[ty][k] * shared_w[k][tx];
		}

		__syncthreads();
	}

	//Attenzione alla divergenza dei threads (vedi CCC pag.137)
	if (dest_x < col_w && dest_y < num_pattern) {

		h2h_dest[dest_y*col_w + dest_x] = 1.0 / (1.0 + (float)exp(-(partial + biases[dest_x]))); //SIGMA
	}
}

/*patt_per_step � il numero di pattern (quando possibile...) da considerare in ciascuna iterazione su h2h*/
/*Questo kernel ad ogni passo incrementa il puntatore ad h2h di num_patt_per_step*NEURO_L_L_1 (e similmente h2h_dest),
controlla che sia ancora nel range di h2h, e calcola num_pattern (vedi sopra) in funzione dei
pattern mancanti*/
//Dove ora c'� STREAMSIZE prima c'era TOTAL_PATT
void MMMulDev(DATA *h2h, DATA *w, DATA *biases, DATA *h2h_dest, unsigned int row_w, unsigned int col_w, unsigned int patt_per_step, dim3 grid, dim3 block, cudaStream_t stream) {

	unsigned int current_patts;
	unsigned int remaining_patts;
	//const int pos_block_y = blockIdx.y*blockDim.x; //Posizione del blocco corrente rispetto alla griglia lungo le y
												   //Assumiamo che i blocchi siano quadrati (blockDim.x = blockDim.y)			
	for (unsigned int x = 0; x < STREAMSIZE; x += patt_per_step) {

		remaining_patts = STREAMSIZE - x;
		current_patts = (remaining_patts < patt_per_step) ? (remaining_patts) : (patt_per_step);

	//	if (pos_block_y >= current_patts) { return; }

		MMMulDevPartial<<< grid, block, 0, stream >>>(h2h + x*row_w, w, biases, h2h_dest + x*col_w, row_w, col_w, current_patts);
	}
}


/*HOST*/

/*FIRT PHASE OF THE ALGORITHM -- THE INPUT IS TRANSMITTED VIA THE NETWORK*/
void feedforward(DATA *INPUT, struct host_to_dev_mem * htdm, struct dev_host_to_dev_mem *dev_htdm, int *nupl, int layers) {
	//cudaEvent_t start, stop;

	cudaStream_t streams[NSTREAMS];
	for (int i = 0; i < NSTREAMS; i++) {
		HANDLE_CUDA(cudaStreamCreate(&streams[i]));
	}

	//Grid setting
	dim3 grid, block;
	unsigned int patt_per_step;

	//startTimer(&start, &stop);
	HANDLE_CUDA(cudaMemcpy(dev_htdm, htdm ,(GLOBAL_BIAS_SIZE + GLOBAL_W_SIZE)*sizeof(DATA),cudaMemcpyHostToDevice));
	//stopAndPrint(&start, &stop);
	
	for (int i = 0; i < NSTREAMS; i++) {

		block.x = gs.block[0];
		block.y = gs.block[0];
		grid.x = (nupl[1] + block.x - 1) / block.x;
		grid.y = gs.grid[0] / grid.x;

		patt_per_step = grid.y * block.y;

		int offset = i*STREAMSIZE;
		//HANDLE_CUDA(cudaMemcpyAsync(dev_H2H[0] + offset*nupl[0], INPUT + offset*nupl[0], nupl[0] * STREAMSIZE * sizeof(DATA), cudaMemcpyHostToDevice, streams[i]));
		//MMMulDev(dev_H2H[0] + offset*nupl[0], dev_WeightH2H[0], dev_BIASH2H[0], dev_H2H[1] + offset*nupl[1], nupl[0], nupl[1], patt_per_step, grid, block, streams[i]);
		HANDLE_CUDA(cudaMemcpyAsync(dev_htdm->H2H + offset*nupl[0], INPUT + offset*nupl[0], nupl[0] * STREAMSIZE * sizeof(DATA), cudaMemcpyHostToDevice, streams[i]));
		MMMulDev(dev_htdm->H2H + offset*nupl[0], dev_htdm->WeightH2H, dev_htdm->BiasH2H, dev_htdm->H2H + htdm->matrix_H2H_index[0][1] + offset*nupl[1], nupl[0], nupl[1], patt_per_step, grid, block, streams[i]);
		
		for (int l = 1; l < (layers - 1); l++) {

			block.x = gs.block[l];
			block.y = gs.block[l];
			grid.x = (nupl[l + 1] + block.x - 1) / block.x;
			grid.y = gs.grid[l] / grid.x;

			patt_per_step = grid.y * block.y;


			int offset = i*STREAMSIZE;
			MMMulDev(dev_htdm->H2H + htdm->matrix_H2H_index[0][l] + offset*nupl[l], dev_htdm->WeightH2H + htdm->matrix_WB_index[0][l], dev_htdm->BiasH2H + htdm->matrix_WB_index[1][l], dev_htdm->H2H + htdm->matrix_H2H_index[0][l+1] + offset*nupl[l+1], nupl[l], nupl[l+1], patt_per_step, grid, block, streams[i]);

			//MMMulDev(dev_H2H[l] + offset*nupl[l], dev_WeightH2H[l], dev_BIASH2H[l], dev_H2H[l + 1] + offset*nupl[l + 1], nupl[l], nupl[l + 1], patt_per_step, grid, block, streams[i]);
			
		}
	}	
	/*
	cudaDeviceSynchronize();
	DATA **H2H_RES = (DATA**)malloc(TOTAL_LAYER * sizeof(DATA*));
	for (int i = 0; i < TOTAL_LAYER; i++) {
		H2H_RES[i] = (DATA*)malloc(TOTAL_PATT*nupl[i] * sizeof(DATA));
	}	
	for (int l = 0; l < (layers - 1); l++) {
		
		HANDLE_CUDA(cudaMemcpy(htdm->H2H+ htdm->matrix_H2H_index[0][l+1],dev_htdm->H2H + htdm->matrix_H2H_index[0][l+1], (TOTAL_PATT)* nupl[l+1] * sizeof(DATA), cudaMemcpyDeviceToHost));
		MMMulHost( htdm->H2H + htdm->matrix_H2H_index[0][l], htdm->WeightH2H + htdm->matrix_WB_index[0][l] , htdm->BiasH2H + htdm->matrix_WB_index[1][l], H2H_RES[l + 1], TOTAL_PATT, nupl[l], nupl[l + 1]);

		BOOL b = matsAreEquals(htdm->H2H+ htdm->matrix_H2H_index[0][l+1], H2H_RES[l + 1], TOTAL_PATT, nupl[l + 1]);
		printf("layer%d %d\n",l, b);
	}*/
}

/*UTILITY FUNCTIONS*/

void HOST_feedforward(DATA *INPUT, DATA **W, DATA **BIAS, DATA **H2H, int *nupl) {

	MMMulHost(INPUT, W[0], BIAS[0], H2H[0], TOTAL_PATT, nupl[0], nupl[1]);
	MMMulHost(H2H[0], W[1], BIAS[1], H2H[1], TOTAL_PATT, nupl[1], nupl[2]);
	MMMulHost(H2H[1], W[2], BIAS[2], H2H[2], TOTAL_PATT, nupl[2], nupl[3]);

}

/*Print a matrix*/
void printMat(DATA *mat, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		printf("ROW %d : {", i);
		for (int j = 0; j < cols; j++) {
			printf("%f - ", mat[i*cols + j]);
		}
		printf("}");
		printf("\n\n");
	}
	printf("\n\n");
}

/*On host multiplication*/
void MMMulHost(DATA *H2H, DATA *W, DATA *BIAS, DATA *H2H_RES, int row_H2H, int col_H2H, int col_W) {

	for (int i = 0; i < row_H2H; i++) {
		for (int j = 0; j < col_W; j++) {
			DATA prod = 0.0;
			for (int k = 0; k < col_H2H; k++) {
				prod += H2H[i*col_H2H + k] * W[k*col_W + j];
			}
			H2H_RES[i*col_W + j] = 1.0 / (1.0 + (float)exp(-(prod + BIAS[j]))); // bias added
		}
	}
}

/*Check device*/
BOOL matsAreEquals(DATA *A, DATA *B, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) { // the first column is for adapting the data
			float err = fabs(A[i*cols + j] - B[i*cols + j]);
			//printf("Error in i=%d,j=%d: %f\n", i, j, err);
			if (err >= MAX_ERR) { printf("row: %d, col: %d\n", i, j); return 0; }
		}
	}
	return 1;
}

/*ALLOCATION FUNCTIONS*/

/* init struct on host */
void HOST_init_struct(struct host_to_dev_mem* htdm, int* nupl) {
		
	int prev_sum[MATRIX_NUMBER_STRUCT];
	htdm->matrix_H2H_index[0][0] = 0;
	htdm->matrix_WB_index[0][0] = 0 ;
	htdm->matrix_WB_index[1][0] = 0;
	//Bisogner� inserire i controlli sulle malloc
	/*il padding della matrice al layer corrente dipende da quello dei layer precedenti*/

	for (unsigned int layer = 1; layer<(TOTAL_LAYER - 1); layer++) {
		
		prev_sum[0] = htdm->matrix_H2H_index[0][layer-1];
		prev_sum[1] = htdm->matrix_WB_index[0][layer-1];
		prev_sum[2] = htdm->matrix_WB_index[1][layer-1];
		
		htdm->matrix_H2H_index[0][layer] = nupl[layer-1] * TOTAL_PATT + prev_sum[0];
		htdm->matrix_WB_index[0][layer] = nupl[layer-1] * nupl[layer] + prev_sum[1];
		htdm->matrix_WB_index[1][layer] = nupl[layer] + prev_sum[2];
		
		for (int i = 0; i < nupl[layer]; i++) {
			for (int j = 0; j < nupl[layer+1]; j++) {
				htdm->WeightH2H[htdm->matrix_WB_index[0][layer] + i*nupl[layer+1] + j] = rand() / (DATA)RAND_MAX;
				htdm->BiasH2H[htdm->matrix_WB_index[1][layer] + j] = rand() / (DATA)RAND_MAX;
			}
		}
		
	}
	for (int i = 0; i < nupl[0]; i++) {
		for (int j = 0; j < nupl[1]; j++) {
				htdm->WeightH2H[i*nupl[1] + j] = rand() / (DATA)RAND_MAX;
				htdm->BiasH2H[j] = rand() / (DATA)RAND_MAX;
		}
	}
}

