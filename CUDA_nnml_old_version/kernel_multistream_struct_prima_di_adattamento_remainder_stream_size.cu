#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DATA float
#define BOOL int
#define MAX_ERR (float)1e-5

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

//Grid features
//Leggere 15 febbraio del diario (passo 1 del feedforward, considerazioni)

#define OPTIMUM_BLOCK_NUM 4 
#define BLOCK_SIDE	16 

#define OPTIMUM_BLOCK_NUM_FIRST_LAYER 2
#define BLOCK_SIDE_FIRST_LAYER 32

/*Struct Grid Settings*/

typedef struct grid_settings {
	int grid[3];
	int block[3];
}grid_settings;

grid_settings gs = { { OPTIMUM_BLOCK_NUM_FIRST_LAYER, OPTIMUM_BLOCK_NUM, OPTIMUM_BLOCK_NUM },{ BLOCK_SIDE_FIRST_LAYER,BLOCK_SIDE,BLOCK_SIDE } };

//Network features

#define NEURO_INPUT 784 //#neurons of input layer
#define NEURO_H_0	56	//#neurons of first hidden layer
#define NEURO_H_1	28	//#neurons of second hidden layer
#define NEURO_OUTPUT 10 //#neurons of output layer
#define TOTAL_PATT	60000 //#total patterns
#define NUM_HIDDEN 2 //#hidden layers
#define TOTAL_LAYER 4 //#of layers

//Streams Settings
#define NSTREAMS 3
#define STREAMSIZE TOTAL_PATT/NSTREAMS

/*Struct One Copy HostToDev -- Contains weights and bias*/

//struct features
#define MATRIX_NUMBER_STRUCT 4 //#matrix to copy to Device(in struct)
#define GLOBAL_H_SIZE TOTAL_PATT * (NEURO_INPUT + NEURO_H_0 + NEURO_H_1 + NEURO_OUTPUT)
#define GLOBAL_DELTA_SIZE TOTAL_PATT * (NEURO_H_0 + NEURO_H_1 + NEURO_OUTPUT)
#define GLOBAL_W_SIZE (NEURO_INPUT*NEURO_H_0) + (NEURO_H_0*NEURO_H_1) + (NEURO_H_1*NEURO_OUTPUT)
#define GLOBAL_BIAS_SIZE NEURO_H_0 + NEURO_H_1 + NEURO_OUTPUT

typedef struct host_to_dev_mem {
	DATA WeightH2H[GLOBAL_W_SIZE];
	DATA BiasH2H[GLOBAL_BIAS_SIZE];
	DATA Delta[GLOBAL_DELTA_SIZE];
	DATA H2H[GLOBAL_H_SIZE];
	int matrix_WB_index[MATRIX_NUMBER_STRUCT - 2][TOTAL_LAYER - 1];//INDEX for padding in Weight & Bias 
	int matrix_DELTA_index[MATRIX_NUMBER_STRUCT - 3][TOTAL_LAYER - 1];//INDEX for padding in Delta
	int matrix_H2H_index[MATRIX_NUMBER_STRUCT - 3][TOTAL_LAYER];//INDEX for padding in H2H
} host_to_dev_mem;

typedef struct dev_struct {
	DATA WeightH2H[GLOBAL_W_SIZE];
	DATA BiasH2H[GLOBAL_BIAS_SIZE];
	DATA Delta[GLOBAL_DELTA_SIZE];
	DATA H2H[GLOBAL_H_SIZE];
} dev_struct;

//Texture reference (FOR TARGET MATRIX)
texture<DATA, 2, cudaReadModeElementType> texreference_target;

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

/*deviceReduceBlockAtomicKernel*/
__inline__ __device__ DATA warpReduceSum(DATA);
__inline__ __device__ DATA blockReduceSum(DATA);
__global__ void deviceReduceBlockAtomicKernel(DATA *, DATA*, int);

/*MMMul(for feedforward)*/
__device__ void MMMulDevPartialFeed(DATA *, DATA *, DATA *, DATA *, DATA*, DATA *, int, int, int, int);
__global__ void MMMulDevFeed(DATA *, DATA *, DATA *, DATA *, DATA *, DATA*, int, int, int, int);
/*MMMul(for backpropagation)*/
__device__ void MMMulDevPartialBack(DATA *, DATA *, DATA *, DATA *, int, int, int);
__global__ void MMMulDevBack(DATA *, DATA *, DATA *, DATA *, int, int, int);

/*HOST*/
void FeedAndBack(DATA *, struct host_to_dev_mem *, struct dev_struct *, DATA *, DATA *, int *, int, cudaStream_t *, BOOL);

void HOST_feedforward(DATA *, DATA **, DATA **, DATA **, int *);
void printMat(DATA *, int, int);
void printErrorMat(DATA *, DATA*, int, int);
void MMMulHost(DATA *, DATA *, DATA *, DATA *, int, int, int);
BOOL matsAreEquals(DATA *, DATA *, int, int);
DATA errorReductionHost(DATA *, int, int);

/*HOST ALLOCATION AND INITIALIZATION*/
void HOST_init_struct(struct host_to_dev_mem*, int*, int);

/*----------------------------------------------------------------------MAIN---------------------------------------------------------------------------*/

int main(void) {

	DATA *INPUT_MAT, *ERROR_MAT, *DEV_ERROR_MAT;
	DATA *ERROR, *DEV_ERROR;
	DATA *TARGET;
	cudaStream_t streams[NSTREAMS];

	int *nupl = (int*)malloc(TOTAL_LAYER * sizeof(int));

	/*++++------------------------------------ERRORS--------------------------------------------------++++*/

	ERROR_MAT = (DATA*)malloc(TOTAL_PATT*NEURO_OUTPUT * sizeof(DATA)); // ERROR FOR CHECKING CORRECTNESS
	HANDLE_CUDA(cudaMalloc((void **)&DEV_ERROR_MAT, TOTAL_PATT*NEURO_OUTPUT * sizeof(DATA))); //DEVICE ERROR MAT

	ERROR = (DATA*)malloc(sizeof(DATA)); // ERROR FOR CHECKING CORRECTNESS
	HANDLE_CUDA(cudaMalloc((void **)&DEV_ERROR, sizeof(DATA))); //DEVICE ERROR
	HANDLE_CUDA(cudaMemset(DEV_ERROR, 0, sizeof(DATA)));

	/*----------------------------------------ERRORS END--------------------------------------------------*/

	/*++++---------------------------init INPUT_MAT and TARGET (HOST)-----------------------------++++*/
	nupl[0] = NEURO_INPUT;
	nupl[1] = NEURO_H_0;
	nupl[2] = NEURO_H_1;
	nupl[TOTAL_LAYER - 1] = NEURO_OUTPUT;

	TARGET = (DATA*)malloc(NEURO_OUTPUT*TOTAL_PATT * sizeof(DATA)); //TARGET OF THE PATTERNS

	for (int i = 0; i < TOTAL_PATT; i++) {
		for (int j = 0; j < NEURO_OUTPUT; j++) {
			TARGET[i*NEURO_OUTPUT + j] = (DATA)rand() / (DATA)RAND_MAX;
		}
	}

	/*INPUT_MAT is pinned memory*/
	
	HANDLE_CUDA(cudaHostAlloc(&INPUT_MAT, NEURO_INPUT * TOTAL_PATT * sizeof(DATA), 0));
	//DATA r;
	for (int i = 0; i < TOTAL_PATT; i++) {
		for (int j = 0; j < NEURO_INPUT; j++) {
			//r= rand() / (DATA)RAND_MAX;
			INPUT_MAT[i*NEURO_INPUT + j] = (DATA)rand() / (DATA)RAND_MAX;
			//htdm->H2H[i*NEURO_INPUT+ j] = r;
		}
	}

	/*---------------------------end init INPUT_MAT and TARGET (HOST)-------------------------*/

	/*++++---------------------------data structures on host and device-------------------------++++*/
	struct host_to_dev_mem *htdm = (struct host_to_dev_mem*)malloc(sizeof(struct host_to_dev_mem));
	struct dev_struct *dev_htdm;

	//Init weights and biases on host
	HOST_init_struct(htdm, nupl, TOTAL_LAYER);
	//Malloc the necessary space on device memory
	HANDLE_CUDA(cudaMalloc((void **)&dev_htdm, sizeof(struct dev_struct)));

	/*---------------------------end data structures on host and device----------------------------*/

	/*++++---------------------------cuda array for texture-----------------------------++++*/
	cudaArray* DEV_TARGET_CUDA;
	cudaChannelFormatDesc channel;

	channel = cudaCreateChannelDesc<DATA>();
	HANDLE_CUDA(cudaMallocArray(&DEV_TARGET_CUDA, &channel, NEURO_OUTPUT, TOTAL_PATT));
	HANDLE_CUDA(cudaMemcpyToArray(DEV_TARGET_CUDA, 0, 0, TARGET, NEURO_OUTPUT*TOTAL_PATT * sizeof(DATA), cudaMemcpyHostToDevice));

	texreference_target.filterMode = cudaFilterModePoint; //turn off the interpolation of cudaFilterModeLinear
	texreference_target.addressMode[0] = cudaAddressModeWrap;//works in normalized coordinates only
	texreference_target.addressMode[1] = cudaAddressModeClamp;//works in both unnormalized and normalized coordinates

	HANDLE_CUDA(cudaBindTextureToArray(texreference_target, DEV_TARGET_CUDA)); //Texture reference binding
	/*---------------------------end cuda array for texture-------------------------*/

	/*++++-----------Streams creation------------++++*/
	for (int i = 0; i < NSTREAMS; i++) {
		HANDLE_CUDA(cudaStreamCreate(&streams[i]));
	}
	/*---------------end--streams creation-----------*/

	/*++++-----------------------------------FEEDFORWARD---AND---BACKPROPAGATION-------------------------------------------++++*/

	cudaEvent_t start, stop;

	startTimer(&start, &stop);
	FeedAndBack(INPUT_MAT, htdm, dev_htdm, DEV_ERROR_MAT, DEV_ERROR, nupl, TOTAL_LAYER, streams, 1);
	stopAndPrint(&start, &stop);
	//cudaDeviceSynchronize();//
	
	HANDLE_CUDA(cudaMemcpy(ERROR, DEV_ERROR, sizeof(DATA), cudaMemcpyDeviceToHost));
	printf("Reduced Error: %f\n", *ERROR);
	
	/*
	HANDLE_CUDA(cudaMemcpy(ERROR_MAT, DEV_ERROR_MAT, TOTAL_PATT*NEURO_OUTPUT * sizeof(DATA), cudaMemcpyDeviceToHost));
	printMat(ERROR_MAT, TOTAL_PATT, NEURO_OUTPUT);
	DATA red_host = errorReductionHost(ERROR_MAT, TOTAL_PATT, NEURO_OUTPUT);
	printf("host reduction error : %f\n", red_host);
	*/

	/*-------------------------------------END---FEEDFORWARD---AND---BACKPROPAGATION-------------------------------------------*/

	/*++++--------------------------------deallocations------------------------------------++++*/
	//Host dealloc
	free(nupl);
	free(htdm);
	free(TARGET);
	free(ERROR_MAT);
	free(ERROR);
	cudaFree(dev_htdm);
	cudaFree(DEV_ERROR_MAT);
	cudaFree(DEV_ERROR);
	cudaFreeHost(INPUT_MAT);
	//Unbinding texture
	cudaUnbindTexture(texreference_target);
	//Free cuda array
	cudaFreeArray(DEV_TARGET_CUDA);

	/*------------------------------------end deallocations------------------------------------*/

	return 0;
}


/*---------------------------------------------------------------------KERNEL--------------------------------------------------------------------------*/

/*DEVICE*/

/*++++---------------------------deviceReduceBlockAtomicKernel---------------------------++++*/

/*Warp reduction*/
__inline__ __device__ DATA warpReduceSum(DATA val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}

/*Block reduction*/
__inline__ __device__ DATA blockReduceSum(DATA val) {

	static __shared__ DATA shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSum(val);

	if (lane == 0) shared[wid] = val;

	__syncthreads();

	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid == 0) val = warpReduceSum(val);

	return val;
}

/*Reducing large arrays--Blocks implementation*/

//Nella chiamata di questo kernel è meglio usare una griglia lineare di 8 blocchi con 256 threads ciascuno -- 
//In tal modo vengono limitati gli accessi alla shared memory rispetto all'implementazione con 2 blocchi da 1024 threads ciascuno
//Attenzione ai possibili arrotondamenti di numeri a virgola mobile dovuti alle atomicAdd.
__global__ void deviceReduceBlockAtomicKernel(DATA *in, DATA* out, int N) {
	DATA sum = 0.0f;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x ; i < N ; i += blockDim.x * gridDim.x) {
		sum += in[i];
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0)
		atomicAdd(out, sum);
}

/*-------------------------------end--deviceReduceBlockAtomicKernel--------------------------*/

/*++++---------------------------MMMul--Feedforward-------------------------++++*/

/* h2h è il puntatore alla porzione dell'h2h globale da considerare in questa fase
(ad ogni passo il kernel che invoca questo device incrementa il puntatore h2h
in modo proporzionale al patt_per_step (e similmente h2h_dest) (vedi sotto)).
offset_y è la posizione considerata lungo le y (nelle matrici h2h, h2h_dest ed eventualmente error) durante la chiamata corrente a __device__.
Delta è calcolato per l'output layer (propagato poi con backpropagation) --> DeltaO[p][k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;
*/

__device__ void MMMulDevPartialFeed(DATA *h2h, DATA *w, DATA *biases, DATA *h2h_dest, DATA *delta, DATA *error, int row_w, int col_w, int num_pattern, int offset_y) {

	int tx = threadIdx.x, ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int block_dim = blockDim.x; // assumiamo che i blocchi siano quadrati
	int dest_x = block_x*block_dim + tx;
	int dest_y = block_y*block_dim + ty;

	int w_x = block_x*block_dim; // start block in w
	int h2h_y = block_y*block_dim*row_w; // start block in h2h

	int end_h2h = h2h_y + row_w - 1; // last block position in h2h

	int step_w = block_dim*col_w;
	int step_h2h = block_dim;

	DATA partial = 0.0f;
	int block_r_border = 0; // contatore che indica in che iterazione dei blocchi ci troviamo
	int current_inc;
	int min;

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

		min = MIN(current_inc, block_dim);

		#pragma unroll(2)
		for (int k = 0; k < min; k++) {
			partial += shared_h2h[ty][k] * shared_w[k][tx];
		}

		__syncthreads();
	}

	//Attenzione alla divergenza dei threads (vedi CCC pag.137)
	if (dest_x < col_w && dest_y < num_pattern) {

		DATA out = (DATA)1.0 / (DATA)(1.0 + exp(-(partial + biases[dest_x])));
		h2h_dest[dest_y*col_w + dest_x] = out; //SIGMA

		//Se siamo nell'ultimo passo
		if (col_w == NEURO_OUTPUT) {
			
			DATA target = tex2D(texreference_target, dest_x, dest_y + offset_y);

			//Scrivi nella posizione corrispondente della matrice di ERRORE
			/*0.5*(Target[p][k] - Output[p][k])*(Target[p][k] - Output[p][k])*/
			error[dest_y*col_w + dest_x] = 0.5*(target - out)*(target - out);

			//Scrivi nella posizione corrispondente della matrice DELTA
			/*(Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k])*/
			delta[dest_y*col_w + dest_x] = (target - out)*(out)*(1 - out);
		}
	}
}

/*patt_per_step è il numero di pattern (quando possibile...) da considerare in ciascuna iterazione su h2h*/
/*Questo kernel ad ogni passo incrementa il puntatore ad h2h di num_patt_per_step*NEURO_L_L_1 (e similmente h2h_dest),
controlla che sia ancora nel range di h2h, e calcola num_pattern (vedi sopra) in funzione dei
pattern mancanti.
stream_offset_y è la posizione lungo le y da cui parte (nelle matrici h2h e h2h_dest) lo stream corrente.
*/
//Dove ora c'è STREAMSIZE prima c'era TOTAL_PATT
__global__ void MMMulDevFeed(DATA *h2h, DATA *w, DATA *biases, DATA *h2h_dest, DATA *delta, DATA *error, int row_w, int col_w, int patt_per_step, int stream_offset_y) {

	int current_patts;
	int remaining_patts;
	int pos_block_y = blockIdx.y*blockDim.x; //Posizione del blocco corrente rispetto alla griglia lungo le y
												      //Assumiamo che i blocchi siano quadrati (blockDim.x = blockDim.y)		

	for (int y = 0; y < STREAMSIZE; y += patt_per_step) {

		remaining_patts = STREAMSIZE - y;
		current_patts = MIN(remaining_patts, patt_per_step);

		if (pos_block_y >= current_patts) { return; }

		MMMulDevPartialFeed(h2h + y*row_w, w, biases, h2h_dest + y*col_w, delta + y*NEURO_OUTPUT, error + y*NEURO_OUTPUT, row_w, col_w, current_patts, stream_offset_y + y);
	}
}

/*-------------------------------end--MMMul--Feedforward------------------------*/

/*++++---------------------------MMMul--BackPropagation-------------------------++++*/

__device__ void MMMulDevPartialBack(DATA *delta_l, DATA *w, DATA *delta_l_1, DATA *h2h_l_1, int row_w, int col_w, int num_pattern) {

	int tx = threadIdx.x, ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int block_dim = blockDim.x; // assumiamo che i blocchi siano quadrati
	int dest_x = block_x*block_dim + tx;
	int dest_y = block_y*block_dim + ty;

	//Dobbiamo scorrere la matrice w per righe (stiamo considerando w come fosse trasposta -- vedi 16 febbraio su diario)
	int w_y = block_x*block_dim*col_w; // start block in w
	int delta_l_y = block_y*block_dim*col_w; // start block in delta_l_y

	int end_delta_l = delta_l_y + col_w - 1; // last block position in h2h
	
	int step_w_y = block_dim;
	int step_delta_l = block_dim;

	DATA partial = 0.0f;
	int block_r_border = 0; // contatore che indica in che iterazione dei blocchi ci troviamo
	int current_inc;
	int min;

	for (int wid = w_y, deltaid = delta_l_y; deltaid <= end_delta_l; wid += step_w_y, deltaid += step_delta_l) {

		block_r_border += block_dim;

		__shared__ DATA shared_w[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER];
		__shared__ DATA shared_delta_l[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER];

		int t_index_w = wid + tx + ty*col_w;
		int t_index_delta_l = deltaid + tx + ty*col_w;

		//Attenzione alla divergenza dei threads (vedi CCC pag.137)
		shared_delta_l[ty][tx] = (t_index_delta_l < num_pattern*col_w) ? (delta_l[t_index_delta_l]) : (0.0f);
		//Salviamo la sottomatrice trasposta nella shared memory della matrice dei pesi (osservare che in tal modo evitiamo conflitti di banco):
		shared_w[tx][ty] = (t_index_w < row_w*col_w) ? (w[t_index_w]) : (0.0f);

		__syncthreads();

		current_inc = col_w - (block_r_border - block_dim);

		min = MIN(current_inc, block_dim);

		#pragma unroll(2)
		for (int k = 0; k < min; k++) {
			partial += shared_delta_l[ty][k] * shared_w[k][tx];
		}

		__syncthreads();
	}

	//Attenzione alla divergenza dei threads (vedi CCC pag.137)
	if (dest_x < row_w && dest_y < num_pattern) {
		//Backpropagate the delta to the previous layer
		DATA h2h_l_1_target = h2h_l_1[dest_y*row_w + dest_x];
		delta_l_1[dest_y*row_w + dest_x] = partial*h2h_l_1_target*(1 - h2h_l_1_target);
	}
}

__global__ void MMMulDevBack(DATA *delta_l, DATA *w, DATA *delta_l_1, DATA *h2h_l_1, int row_w, int col_w, int patt_per_step) {

	int current_patts;
	int remaining_patts;
	int pos_block_y = blockIdx.y*blockDim.x; //Posizione del blocco corrente rispetto alla griglia lungo le y
												      //Assumiamo che i blocchi siano quadrati (blockDim.x = blockDim.y)		

	for (int y = 0; y < STREAMSIZE; y += patt_per_step) {

		remaining_patts = STREAMSIZE - y;
		current_patts = MIN(remaining_patts, patt_per_step);

		if (pos_block_y >= current_patts) { return; }

		MMMulDevPartialBack(delta_l + y*col_w, w, delta_l_1 + y*row_w, h2h_l_1 + y*row_w, row_w, col_w, current_patts);
	}
}

/*-------------------------------end--MMMul--BackPropagation------------------------*/

/*HOST*/

/*FEEDFORWARD AND BACKPROPAGATION PHASES -- THE INPUT IS TRANSMITTED VIA THE NETWORK AND IN BACK PROPAGATED*/
void FeedAndBack(DATA *INPUT, struct host_to_dev_mem * htdm, struct dev_struct *dev_htdm, DATA *dev_error_mat, DATA *dev_error, int *nupl, int layers, cudaStream_t *streams, BOOL first_epoch) {
	//cudaEvent_t start, stop;

	//Grid setting
	dim3 grid, block;
	int patt_per_step;
	//Useful pointers
	DATA *h2h, *w, *bias, *h2h_dest, *delta, *error;
	//Delta da cui parte l'informazione (delta_l) e delta dove arriva tramite la backpropagation (delta_l_1)
	DATA *delta_l, *delta_l_1; 

	//offset
	int offset;

	//startTimer(&start, &stop);
	if (first_epoch) {
		HANDLE_CUDA(cudaMemcpy(dev_htdm, htdm, (GLOBAL_BIAS_SIZE + GLOBAL_W_SIZE) * sizeof(DATA), cudaMemcpyHostToDevice));
	}
	//stopAndPrint(&start, &stop);

	for (int i = 0; i < NSTREAMS; i++) {

		//Leggere 15 febbraio del diario (passo 1 del feedforward, considerazioni)
		block.x = gs.block[0];
		block.y = gs.block[0];
		grid.x = (nupl[1] + block.x - 1) / block.x;
		grid.y = MAX(gs.grid[0] / grid.x, 1); //Evitare che possa diventare 0

		patt_per_step = grid.y * block.y;

		offset = i*STREAMSIZE;
		//Set pointers
		h2h = dev_htdm->H2H + offset*nupl[0];
		w = dev_htdm->WeightH2H;
		bias = dev_htdm->BiasH2H;
		h2h_dest = dev_htdm->H2H + htdm->matrix_H2H_index[0][1] + offset*nupl[1];
		delta = dev_htdm->Delta + htdm->matrix_DELTA_index[0][layers - 2] + offset*nupl[layers - 1];
		error = dev_error_mat + offset*nupl[layers - 1];
		//Pointers set up

		if (first_epoch) {
			HANDLE_CUDA(cudaMemcpyAsync(h2h, INPUT + offset*nupl[0], nupl[0] * STREAMSIZE * sizeof(DATA), cudaMemcpyHostToDevice, streams[i]));
		}

		//First Feedforward step:
		MMMulDevFeed << <grid, block, 0, streams[i] >> > (h2h, w, bias, h2h_dest, delta, error, nupl[0], nupl[1], patt_per_step, offset);

		for (int l = 1; l < (layers - 2); l++) {

			block.x = gs.block[l];
			block.y = gs.block[l];
			grid.x = (nupl[l + 1] + block.x - 1) / block.x;
			grid.y = MAX(gs.grid[l] / grid.x, 1); //Evitare che possa diventare 0

			patt_per_step = grid.y * block.y;
			//Set pointers
			h2h = dev_htdm->H2H + htdm->matrix_H2H_index[0][l] + offset*nupl[l];
			w = dev_htdm->WeightH2H + htdm->matrix_WB_index[0][l];
			bias = dev_htdm->BiasH2H + htdm->matrix_WB_index[1][l];
			h2h_dest = dev_htdm->H2H + htdm->matrix_H2H_index[0][l + 1] + offset*nupl[l + 1];
			//Delta and error already set up
			//Pointers set up

			MMMulDevFeed << <grid, block, 0, streams[i] >> > (h2h, w, bias, h2h_dest, delta, error, nupl[l], nupl[l + 1], patt_per_step, offset);
		}

		//Last Feedforward step:

		block.x = gs.block[layers - 2];
		block.y = gs.block[layers - 2];
		grid.x = (nupl[layers - 1] + block.x - 1) / block.x;
		grid.y = MAX(gs.grid[layers - 2] / grid.x, 1); //Evitare che possa diventare 0

		patt_per_step = grid.y * block.y;
		//Set pointers
		h2h = dev_htdm->H2H + htdm->matrix_H2H_index[0][layers - 2] + offset*nupl[layers - 2];
		w = dev_htdm->WeightH2H + htdm->matrix_WB_index[0][layers - 2];
		bias = dev_htdm->BiasH2H + htdm->matrix_WB_index[1][layers - 2];
		h2h_dest = dev_htdm->H2H + htdm->matrix_H2H_index[0][layers - 1] + offset*nupl[layers - 1];
		//Delta and error already set up
		//Pointers set up

		MMMulDevFeed << <grid, block, 0, streams[i] >> > (h2h, w, bias, h2h_dest, delta, error, nupl[layers - 2], nupl[layers - 1], patt_per_step, offset);

		//BackPropagate for all streams:
		for (int delta_index = (layers - 2); delta_index > 0; delta_index--) {

			block.x = gs.block[delta_index];
			block.y = gs.block[delta_index];
			grid.x = (nupl[delta_index] + block.x - 1) / block.x;
			grid.y = MAX(gs.grid[delta_index] / grid.x, 1); //Evitare che possa diventare 0

			patt_per_step = grid.y * block.y;
			//Set pointers
			h2h = dev_htdm->H2H + htdm->matrix_H2H_index[0][delta_index] + offset*nupl[delta_index];
			w = dev_htdm->WeightH2H + htdm->matrix_WB_index[0][delta_index];
			delta_l = dev_htdm->Delta + htdm->matrix_DELTA_index[0][delta_index] + offset*nupl[delta_index + 1];
			delta_l_1 = dev_htdm->Delta + htdm->matrix_DELTA_index[0][delta_index - 1] + offset*nupl[delta_index];
			//Pointers set up
			MMMulDevBack << <grid, block, 0, streams[i] >> > (delta_l, w, delta_l_1, h2h, nupl[delta_index], nupl[delta_index + 1], patt_per_step);
		}
	}
	//**HERE**
	//Error reduction (default stream)
	deviceReduceBlockAtomicKernel << <OPTIMUM_BLOCK_NUM * 2, BLOCK_SIDE*BLOCK_SIDE >> > (dev_error_mat, dev_error, TOTAL_PATT*nupl[layers-1]);
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

/*Print error matrix on host (for checking correctness of device)*/
void printErrorMat(DATA *TARGET, DATA *OUTPUT_MAT, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		printf("ROW %d : {", i);
		for (int j = 0; j < cols; j++) {
			printf("%f - ", 0.5*(TARGET[i*cols + j] - OUTPUT_MAT[i*cols + j])*(TARGET[i*cols + j] - OUTPUT_MAT[i*cols + j]));
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
			H2H_RES[i*col_W + j] = (DATA)1.0 / (DATA)(1.0 + exp(-(prod + BIAS[j]))); // bias added
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

/*Check device reduction*/
DATA errorReductionHost(DATA *error_mat, int rows, int cols) {

	DATA reduction = 0.0f;

	for (int i = 0; i < rows*cols; i++) {
		reduction += error_mat[i];
	}

	return reduction;
}

/*ALLOCATION FUNCTIONS*/

/*init struct on host*/
void HOST_init_struct(struct host_to_dev_mem* htdm, int* nupl, int layers) {

	int prev_sum[MATRIX_NUMBER_STRUCT];
	htdm->matrix_H2H_index[0][0] = 0;
	htdm->matrix_DELTA_index[0][0] = 0;
	htdm->matrix_WB_index[0][0] = 0;
	htdm->matrix_WB_index[1][0] = 0;
	//Bisognerà inserire i controlli sulle malloc
	/*il padding della matrice al layer corrente dipende da quello dei layer precedenti*/

	for (int layer = 1; layer<(layers - 1); layer++) {

		prev_sum[0] = htdm->matrix_H2H_index[0][layer - 1];
		prev_sum[1] = htdm->matrix_DELTA_index[0][layer - 1];
		prev_sum[2] = htdm->matrix_WB_index[0][layer - 1];
		prev_sum[3] = htdm->matrix_WB_index[1][layer - 1];

		htdm->matrix_H2H_index[0][layer] = nupl[layer - 1] * TOTAL_PATT + prev_sum[0];
		htdm->matrix_DELTA_index[0][layer] = nupl[layer] * TOTAL_PATT + prev_sum[1];
		htdm->matrix_WB_index[0][layer] = nupl[layer - 1] * nupl[layer] + prev_sum[2];
		htdm->matrix_WB_index[1][layer] = nupl[layer] + prev_sum[3];

		for (int i = 0; i < nupl[layer]; i++) {
			for (int j = 0; j < nupl[layer + 1]; j++) {
				htdm->WeightH2H[htdm->matrix_WB_index[0][layer] + i*nupl[layer + 1] + j] = (DATA)rand() / (DATA)RAND_MAX;
				htdm->BiasH2H[htdm->matrix_WB_index[1][layer] + j] = (DATA)rand() / (DATA)RAND_MAX;
			}
		}

	}
	prev_sum[0] = htdm->matrix_H2H_index[0][layers - 2];
	htdm->matrix_H2H_index[0][layers - 1] = nupl[layers - 2] * TOTAL_PATT + prev_sum[0];

	for (int i = 0; i < nupl[0]; i++) {
		for (int j = 0; j < nupl[1]; j++) {
			htdm->WeightH2H[i*nupl[1] + j] = (DATA)rand() / (DATA)RAND_MAX;
			htdm->BiasH2H[j] = (DATA)rand() / (DATA)RAND_MAX;
		}
	}
}

//NON CANCELLARE !!! INSERIRE NEL FEEDFORWARD PER FARE TEST DI CORRETTEZZA NEL PUNTO **HERE**!!!
//RICORDARSI DI DECOMMENTARE LA 'r' NEL MAIN

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