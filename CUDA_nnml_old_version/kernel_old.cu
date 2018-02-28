#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DATA float
#define BOOL int
#define MAX_ERR 1e-5

#define OPTIMUM_BLOCK_NUM 8
#define BLOCK_SIDE	16

#define NEURO_L_L	28	//#neuroni del layer L
#define NEURO_L_L_1	56	//#neuroni del layer L - 1
#define TOTAL_PATT	60000 //#di pattern totali

/*UTILITIES*/

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void startTimer(cudaEvent_t *start, cudaEvent_t *stop) {
	HANDLE_ERROR(cudaEventCreate(start));
	HANDLE_ERROR(cudaEventCreate(stop));
	HANDLE_ERROR(cudaEventRecord(*start, 0));
}

void stopAndPrint(cudaEvent_t *start, cudaEvent_t *stop) {
	HANDLE_ERROR(cudaEventRecord(*stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(*stop));
	float time = 0.0f;
	HANDLE_ERROR(cudaEventElapsedTime(&time, *start, *stop));
	printf("Elapsed Time: %f milliseconds\n", time);
	HANDLE_ERROR(cudaEventDestroy(*start));
	HANDLE_ERROR(cudaEventDestroy(*stop));
}

/*HOST*/

void printMat(DATA **, int, int);
void MMMulHost(DATA **, DATA **, DATA **, int, int, int);
BOOL matsAreEquals(DATA **, DATA **, int, int);

/*DEVICE*/

/* h2h è il puntatore alla porzione dell'h2h globale da considerare in questa fase
(ad ogni passo il kernel che invoca questo device incrementa il puntatore h2h
in modo proporzionale al patt_per_step (e similmente h2h_dest) (vedi sotto))*/

__device__ void MMMulDev(DATA *h2h, DATA *w, DATA *biases, DATA * h2h_dest, unsigned int num_pattern) {

	int tx = threadIdx.x, ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int dest_x = block_x*BLOCK_SIDE + tx;
	int dest_y = block_y*BLOCK_SIDE + ty;

	int w_x = block_x*BLOCK_SIDE; // start block in w
	int h2h_y = block_y*BLOCK_SIDE*NEURO_L_L_1; // start block in h2h

	int end_h2h = h2h_y + NEURO_L_L_1 - 1; // last block position in h2h

	int step_w = BLOCK_SIDE*NEURO_L_L;
	int step_h2h = BLOCK_SIDE;
	int min;

	DATA partial = 0.0f;
	int block_r_border = 0; // contatore che indica in che iterazione dei blocchi ci troviamo
	int current_inc;

	for (int wid = w_x, h2h_id = h2h_y; h2h_id <= end_h2h; wid += step_w, h2h_id += step_h2h) {

		block_r_border += BLOCK_SIDE;

		__shared__ DATA shared_w[BLOCK_SIDE][BLOCK_SIDE];
		__shared__ DATA shared_h2h[BLOCK_SIDE][BLOCK_SIDE];

		int t_index_w = wid + tx + ty*NEURO_L_L;
		int t_index_h2h = h2h_id + tx + ty*NEURO_L_L_1;

		//Attenzione alla divergenza dei threads (vedi CCC pag.137)
		shared_h2h[ty][tx] = (t_index_h2h < num_pattern*NEURO_L_L_1) ? (h2h[t_index_h2h]) : (0.0f);
		shared_w[ty][tx] = (t_index_w < NEURO_L_L*NEURO_L_L_1) ? (w[t_index_w]) : (0.0f);

		__syncthreads();

		current_inc = NEURO_L_L_1 - (block_r_border - BLOCK_SIDE);

		min = (current_inc < BLOCK_SIDE) ? (current_inc) : (BLOCK_SIDE);

		for (int k = 0; k < min; k++) {
			partial += shared_h2h[ty][k] * shared_w[k][tx];
		}

		__syncthreads();
	}

	//Attenzione alla divergenza dei threads (vedi CCC pag.137)
	if (dest_x < NEURO_L_L && dest_y < num_pattern) {

		h2h_dest[dest_y*NEURO_L_L + dest_x] = 1.0 / (1.0 + (float)exp(-(partial + biases[dest_x]))); //SIGMA
	}
}

/*patt_per_step è il numero di pattern (quando possibile...) da considerare in ciascuna iterazione su h2h*/
/*Questo kernel ad ogni passo incrementa il puntatore ad h2h di num_patt_per_step*NEURO_L_L_1 (e similmente h2h_dest),
controlla che sia ancora nel range di h2h, e calcola num_pattern (vedi sopra) in funzione dei
pattern mancanti*/
__global__ void feedforward(DATA *h2h, DATA *w, DATA *biases, DATA *h2h_dest, unsigned int patt_per_step) {

	unsigned int current_patts;
	unsigned int remaining_patts;
	const int pos_block_y = blockIdx.y*BLOCK_SIDE; //Posizione del blocco corrente rispetto alla griglia lungo le y

	for (unsigned int x = 0; x < TOTAL_PATT; x += patt_per_step) {

		remaining_patts = TOTAL_PATT - x;
		current_patts = (remaining_patts < patt_per_step) ? (remaining_patts) : (patt_per_step);

		if (pos_block_y >= current_patts) { return; }

		MMMulDev(h2h + x*NEURO_L_L_1, w, biases, h2h_dest + x*NEURO_L_L, current_patts);
	}
}

int main(void) {

	cudaEvent_t start, stop;

	unsigned int patt_per_step;

	DATA **W = (DATA**)malloc((NEURO_L_L_1 + 1) * sizeof(DATA*));
	DATA **H2H = (DATA**)malloc(TOTAL_PATT * sizeof(DATA*));
	DATA **H2H_res = (DATA**)malloc(TOTAL_PATT * sizeof(DATA*));
	DATA **H2H2_HOST_RES = (DATA**)malloc(TOTAL_PATT * sizeof(DATA*));
	DATA *dev_W, *dev_H2H_res, *dev_H2H, *dev_biases;
	HANDLE_ERROR(cudaMalloc(&dev_W, NEURO_L_L_1*NEURO_L_L * sizeof(DATA)));
	HANDLE_ERROR(cudaMalloc(&dev_H2H_res, TOTAL_PATT*NEURO_L_L * sizeof(DATA)));
	HANDLE_ERROR(cudaMalloc(&dev_H2H, TOTAL_PATT*NEURO_L_L_1 * sizeof(DATA)));
	HANDLE_ERROR(cudaMalloc(&dev_biases, NEURO_L_L * sizeof(DATA)));

	if (W == NULL || H2H == NULL || H2H2_HOST_RES == NULL || H2H_res == NULL || dev_biases == NULL) {
		fprintf(stderr, "Error in host malloc\n");
		exit(1);
	}

	//W mat

	for (int i = 0; i < NEURO_L_L_1 + 1; i++) {
		W[i] = (DATA*)malloc((NEURO_L_L + 1) * sizeof(DATA));

		for (int j = 0; j < NEURO_L_L + 1; j++) {
			W[i][j] = rand() / (DATA)RAND_MAX;
		}

		if (i == 0) {
			HANDLE_ERROR(cudaMemcpy(dev_biases, W[i] + 1, NEURO_L_L * sizeof(DATA), cudaMemcpyHostToDevice));
		}

		else {
			HANDLE_ERROR(cudaMemcpy(dev_W + (i - 1)*NEURO_L_L, W[i] + 1, NEURO_L_L * sizeof(DATA), cudaMemcpyHostToDevice));
		}
	}

	//H2H & H2H_res mat

	for (int i = 0; i < TOTAL_PATT; i++) {
		H2H[i] = (DATA*)malloc((NEURO_L_L_1 + 1) * sizeof(DATA));
		H2H_res[i] = (DATA*)malloc((NEURO_L_L + 1) * sizeof(DATA));
		H2H2_HOST_RES[i] = (DATA*)malloc((NEURO_L_L + 1) * sizeof(DATA));

		for (int j = 0; j < NEURO_L_L_1 + 1; j++) {
			H2H[i][j] = rand() / (DATA)RAND_MAX;
		}
		HANDLE_ERROR(cudaMemcpy(dev_H2H + i*NEURO_L_L_1, H2H[i] + 1, NEURO_L_L_1 * sizeof(DATA), cudaMemcpyHostToDevice));

		for (int j = 0; j < NEURO_L_L + 1; j++) {
			H2H_res[i][j] = 1.0f;
			H2H2_HOST_RES[i][j] = 1.0f;
		}
		HANDLE_ERROR(cudaMemcpy(dev_H2H_res + i*NEURO_L_L, H2H_res[i] + 1, NEURO_L_L * sizeof(DATA), cudaMemcpyHostToDevice));
	}

	//Grid setting
	dim3 grid, block;
	block.x = BLOCK_SIDE;
	block.y = BLOCK_SIDE;
	grid.x = (NEURO_L_L + BLOCK_SIDE - 1) / BLOCK_SIDE;
	grid.y = OPTIMUM_BLOCK_NUM / grid.x;

	patt_per_step = grid.y * 16;


	startTimer(&start, &stop);

	feedforward << <grid, block >> > (dev_H2H, dev_W, dev_biases, dev_H2H_res, patt_per_step);

	stopAndPrint(&start, &stop);


	//Copy result back
	for (int i = 0; i < TOTAL_PATT; i++) {
		HANDLE_ERROR(cudaMemcpy(H2H_res[i] + 1, dev_H2H_res + i*NEURO_L_L, NEURO_L_L * sizeof(DATA), cudaMemcpyDeviceToHost));
	}

	//Print result
	//printMat(H2H_res, TOTAL_PATT, NEURO_L_L);

	//Check the correctness of device computation
	MMMulHost(H2H, W, H2H2_HOST_RES, TOTAL_PATT, NEURO_L_L_1 + 1, NEURO_L_L + 1);

	BOOL b = matsAreEquals(H2H_res, H2H2_HOST_RES, TOTAL_PATT, NEURO_L_L + 1);
	printf("Mats are equals? %d\n", b);

	//printMat(H2H2_HOST_RES, TOTAL_PATT, NEURO_L_L);

	for (int i = 0; i < NEURO_L_L_1 + 1; i++) {
		free(W[i]);
	}

	for (int i = 0; i < TOTAL_PATT; i++) {
		free(H2H[i]);
		free(H2H_res[i]);
		free(H2H2_HOST_RES[i]);
	}

	free(W);
	free(H2H);
	free(H2H2_HOST_RES);
	free(H2H_res);
	HANDLE_ERROR(cudaFree(dev_W));
	HANDLE_ERROR(cudaFree(dev_H2H));
	HANDLE_ERROR(cudaFree(dev_H2H_res));
	HANDLE_ERROR(cudaFree(dev_biases));

	return 0;

}


/*HOST*/

void printMat(DATA **mat, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		printf("ROW %d : {", i);
		for (int j = 0; j < cols; j++) {
			printf("%f - ", mat[i][j]);
		}
		printf("}");
		printf("\n\n");
	}
	printf("\n\n");
}

/*On host multiplication*/
void MMMulHost(DATA **A, DATA **B, DATA **C, int row_a, int col_a, int col_b) {

	for (int i = 0; i < row_a; i++) {
		for (int j = 1; j < col_b; j++) {
			DATA prod = 0.0;
			for (int k = 1; k < col_a; k++) {
				prod += A[i][k] * B[k][j];
			}
			C[i][j] = 1.0 / (1.0 + (float)exp(-(prod + B[0][j]))); // bias added
		}
	}
}

/*Check device*/
BOOL matsAreEquals(DATA **A, DATA **B, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 1; j < cols; j++) { // the first column is for adapting the data
			float err = fabs(A[i][j] - B[i][j]);
			//printf("Error in i=%d,j=%d: %f\n", i, j, err);
			if (err >= MAX_ERR) { printf("row: %d, col: %d\n", i, j); return 0; }
		}
	}
	return 1;
}