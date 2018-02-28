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

/*Struct Grid Settings*/

typedef struct grid_settings {
	unsigned int grid[3];
	unsigned int block[3];
}grid_settings;

grid_settings gs = { { OPTIMUM_BLOCK_NUM_FIRST_LAYER, OPTIMUM_BLOCK_NUM, OPTIMUM_BLOCK_NUM },{BLOCK_SIDE_FIRST_LAYER,BLOCK_SIDE,BLOCK_SIDE} };

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

/*HOST*/
void feedforward(DATA **, DATA **, DATA **, DATA **, DATA **, DATA **, int *, int);
void printMat(DATA *, int, int);
void MMMulHost(DATA *, DATA *, DATA *, DATA *, int, int, int);
BOOL matsAreEquals(DATA *, DATA *, int, int);

/*HOST ALLOCATION AND INITIALIZATION*/

void HOST_alloc_init(DATA** , DATA** , DATA** , int* );

/*DEVICE ALLOCATION*/

void CUDA_alloc(DATA**, DATA**, DATA**, int*);

/*HOST DEALLOCATION FUNCTIONS*/

void HOST_dealloc(DATA**, DATA**, DATA**);

/*DEVICE DEALLOCATION FUNCTIONS*/

void CUDA_dealloc(DATA**, DATA**, DATA**);

/*---------------------------------------------------------------------KERNEL--------------------------------------------------------------------------*/

/*DEVICE*/

/* h2h è il puntatore alla porzione dell'h2h globale da considerare in questa fase
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

		__shared__ DATA shared_w[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER+1];
		__shared__ DATA shared_h2h[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER+1];

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

/*patt_per_step è il numero di pattern (quando possibile...) da considerare in ciascuna iterazione su h2h*/
/*Questo kernel ad ogni passo incrementa il puntatore ad h2h di num_patt_per_step*NEURO_L_L_1 (e similmente h2h_dest),
controlla che sia ancora nel range di h2h, e calcola num_pattern (vedi sopra) in funzione dei
pattern mancanti*/
void MMMulDev(DATA *h2h, DATA *w, DATA *biases, DATA *h2h_dest, unsigned int row_w, unsigned int col_w , unsigned int patt_per_step, dim3 grid, dim3 block) {

	unsigned int current_patts;
	unsigned int remaining_patts;
	//const int pos_block_y = blockIdx.y*blockDim.x; //Posizione del blocco corrente rispetto alla griglia lungo le y
												   //Assumiamo che i blocchi siano quadrati (blockDim.x = blockDim.y)			
	for (unsigned int x = 0; x < TOTAL_PATT; x += patt_per_step) {

		remaining_patts = TOTAL_PATT - x;
		current_patts = (remaining_patts < patt_per_step) ? (remaining_patts) : (patt_per_step);

		//if (pos_block_y >= current_patts) { return; }

		MMMulDevPartial<<<grid,block>>>(h2h + x*row_w, w, biases, h2h_dest + x*col_w, row_w, col_w, current_patts);
	}
}

/*----------------------------------------------------------------------MAIN---------------------------------------------------------------------------*/

int main(void) {

	DATA **H2H_MAT, **W_MAT, **BIAS_MAT, **DEV_H2H_MAT, **DEV_W_MAT, **DEV_BIAS_MAT;
	int *nupl = (int*)malloc(TOTAL_LAYER * sizeof(int));

	/*questa parte bisognerà renderla dinamica. In seguito bisognerà accedere ai files.*/
	nupl[0] = NEURO_INPUT;
	nupl[1] = NEURO_H_0;
	nupl[2] = NEURO_H_1;
	nupl[TOTAL_LAYER - 1] = NEURO_OUTPUT;

	/*host memory allocation and initialization*/
	H2H_MAT = (DATA**)malloc(TOTAL_LAYER * sizeof(DATA*));
	W_MAT = (DATA**)malloc((TOTAL_LAYER - 1) * sizeof(DATA*));
	BIAS_MAT = (DATA**)malloc((TOTAL_LAYER - 1) * sizeof(DATA*));
	HOST_alloc_init(H2H_MAT, W_MAT, BIAS_MAT, nupl);

	/*device memory allocation*/
	DEV_H2H_MAT = (DATA**)malloc(TOTAL_LAYER * sizeof(DATA*));
	DEV_W_MAT = (DATA**)malloc((TOTAL_LAYER - 1) * sizeof(DATA*));
	DEV_BIAS_MAT = (DATA**)malloc((TOTAL_LAYER - 1) * sizeof(DATA*));
	CUDA_alloc(DEV_H2H_MAT, DEV_W_MAT, DEV_BIAS_MAT, nupl);
	/*-----------------------------------FEEDFORWARD-------------------------------------------*/
	//cudaEvent_t start, stop;
	//startTimer(&start, &stop);
	feedforward(H2H_MAT, W_MAT, BIAS_MAT, DEV_H2H_MAT, DEV_W_MAT, DEV_BIAS_MAT, nupl, TOTAL_LAYER);
	//stopAndPrint(&start, &stop);

	/*-----------------------------END---FEEDFORWARD-------------------------------------------*/

	//Host dealloc
	free(nupl);
	HOST_dealloc(H2H_MAT, W_MAT, BIAS_MAT);
	//Cuda dealloc
	CUDA_dealloc(DEV_H2H_MAT, DEV_W_MAT, DEV_BIAS_MAT);
	free(DEV_H2H_MAT);
	free(DEV_W_MAT);
	free(DEV_BIAS_MAT);

	return 0;
}

/*HOST*/

/*FIRT PHASE OF THE ALGORITHM -- THE INPUT IS TRANSMITTED VIA THE NETWORK*/
void feedforward(DATA **H2H, DATA **WeightH2H, DATA **BiasH2H, DATA **dev_H2H, DATA **dev_WeightH2H, DATA **dev_BIASH2H, int *nupl, int layers) {

	cudaEvent_t start, stop;

	//Grid setting
	dim3 grid, block;
	unsigned int patt_per_step;
	
	startTimer(&start, &stop);
	HANDLE_CUDA(cudaMemcpy(dev_H2H[0], H2H[0], (TOTAL_PATT)* nupl[0] * sizeof(DATA), cudaMemcpyHostToDevice));
	stopAndPrint(&start, &stop);

	//loop layers
	for (int l = 0; l < (layers - 1); l++) {

		block.x = gs.block[l];
		block.y = gs.block[l];
		grid.x = (nupl[l+1] + block.x - 1) / block.x;
		grid.y = gs.grid[l] / grid.x;

		patt_per_step = grid.y * block.y;

		//printf("block x = %u block y = %u grid x = %u grid y = %u pattern per step = %u\n", block.x, block.y, grid.x, grid.y, patt_per_step);
		startTimer(&start, &stop);
		HANDLE_CUDA(cudaMemcpy(dev_WeightH2H[l], WeightH2H[l], sizeof(DATA)*nupl[l] * nupl[l+1], cudaMemcpyHostToDevice));
		HANDLE_CUDA(cudaMemcpy(dev_BIASH2H[l], BiasH2H[l], sizeof(DATA)*nupl[l+1], cudaMemcpyHostToDevice));
		stopAndPrint(&start, &stop);

		startTimer(&start, &stop);
		MMMulDev(dev_H2H[l], dev_WeightH2H[l], dev_BIASH2H[l], dev_H2H[l+1], nupl[l], nupl[l+1], patt_per_step,grid,block);
		stopAndPrint(&start, &stop);
		cudaDeviceSynchronize();	
	}
}

/*UTILITY FUNCTIONS*/

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

/*Allocation and initialization of host memory*/
void HOST_alloc_init(DATA** H2H_MAT, DATA** W_MAT, DATA** BIAS_MAT, int* nupl) {

	//Bisognerà inserire i controlli sulle malloc
	for (int layer = 0; layer<(TOTAL_LAYER - 1); layer++) {
		H2H_MAT[layer] = (DATA*)malloc(nupl[layer] * TOTAL_PATT * sizeof(DATA));
		W_MAT[layer] = (DATA*)malloc(nupl[layer] * nupl[layer + 1] * sizeof(DATA));
		BIAS_MAT[layer] = (DATA*)malloc(nupl[layer + 1] * sizeof(DATA));

		for (int i = 0; i < nupl[layer]; i++) {
			for (int j = 0; j < nupl[layer + 1]; j++) {
				W_MAT[layer][i*nupl[layer + 1] + j] = rand() / (DATA)RAND_MAX;
				BIAS_MAT[layer][j] = rand() / (DATA)RAND_MAX;
			}
		}
	}

	H2H_MAT[TOTAL_LAYER - 1] = (DATA*)malloc(nupl[TOTAL_LAYER - 1] * TOTAL_PATT * sizeof(DATA));

	for (int i = 0; i < TOTAL_PATT; i++) {
		for (int j = 0; j < NEURO_INPUT; j++) {
			H2H_MAT[0][i*NEURO_INPUT + j] = rand() / (DATA)RAND_MAX;
		}
	}
}

/*Allocation of device memory (by host)*/
void CUDA_alloc(DATA** DEV_H2H_MAT, DATA** DEV_W_MAT, DATA** DEV_BIAS_MAT, int* nupl) {

	for (int layer = 0; layer<(TOTAL_LAYER - 1); layer++) {
		HANDLE_CUDA(cudaMalloc(&(DEV_H2H_MAT[layer]), nupl[layer] * TOTAL_PATT * sizeof(DATA)));
		HANDLE_CUDA(cudaMalloc(&(DEV_W_MAT[layer]), nupl[layer] * nupl[layer + 1] * sizeof(DATA)));
		HANDLE_CUDA(cudaMalloc(&(DEV_BIAS_MAT[layer]), nupl[layer + 1] * sizeof(DATA)));
	}
	HANDLE_CUDA(cudaMalloc(&(DEV_H2H_MAT[TOTAL_LAYER - 1]), nupl[TOTAL_LAYER - 1] * TOTAL_PATT * sizeof(DATA)));
}

/*DEALLOCATION FUNCTIONS*/

/*Deallocation of host memory*/
void HOST_dealloc(DATA** H2H_MAT, DATA** W_MAT, DATA** BIAS_MAT) {

	for (int layer = 0; layer<(TOTAL_LAYER - 1); layer++) {
		free(H2H_MAT[layer]);
		free(W_MAT[layer]);
		free(BIAS_MAT[layer]);
	}
	free(H2H_MAT[TOTAL_LAYER - 1]);

	free(H2H_MAT);
	free(W_MAT);
	free(BIAS_MAT);
}

/*Deallocation of device memory (called by host)*/
void CUDA_dealloc(DATA** DEV_H2H_MAT, DATA** DEV_W_MAT, DATA** DEV_BIAS_MAT) {
	for (int layer = 0; layer<(TOTAL_LAYER - 1); layer++) {
		cudaFree(DEV_H2H_MAT[layer]);
		cudaFree(DEV_W_MAT[layer]);
		cudaFree(DEV_BIAS_MAT[layer]);
	}
	cudaFree(DEV_H2H_MAT[TOTAL_LAYER - 1]);
}



/*
//Per fare il check della correttezza del prodotto MM sul device:

//questa parte prima del for del feedforward

DATA **H2H_RES = (DATA**)malloc(TOTAL_LAYER * sizeof(DATA*));

for (int i = 0; i < TOTAL_LAYER; i++) {
H2H_RES[i] = (DATA*)malloc(TOTAL_PATT*nupl[i] * sizeof(DATA));
}

//questa parte dentro il for del feedforward dopo la syncdevice

HANDLE_CUDA(cudaMemcpy(H2H[l+1], dev_H2H[l+1], (TOTAL_PATT)* nupl[l+1] * sizeof(DATA), cudaMemcpyDeviceToHost));
MMMulHost(H2H[l], WeightH2H[l], BiasH2H[l], H2H_RES[l + 1], TOTAL_PATT, nupl[l], nupl[l + 1]);
BOOL b = matsAreEquals(H2H_RES[l + 1], H2H[l + 1], TOTAL_PATT, nupl[l + 1]);
printf("%d\n", b);

*/
