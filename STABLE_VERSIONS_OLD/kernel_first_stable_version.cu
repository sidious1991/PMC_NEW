#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DATA float
#define BOOL int
#define MAX_ERR (float)1e-5
#define MAX_EPOCHS 3

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

//Grid features
//Leggere 15 febbraio del diario (passo 1 del feedforward, considerazioni)

#define OPTIMUM_BLOCK_NUM 4 //In vista della concorrenza dei kernels
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

#define NEURO_INPUT 784 //#neuroni dell'input layer
#define NEURO_H_0	56	//#neuroni del primo hidden layer
#define NEURO_H_1	28	//#neuroni del secondo hidden layer
#define NEURO_OUTPUT 10 //#neuroni dell'output layer
#define TOTAL_PATT	60000 //#patterns totali
#define NUM_HIDDEN 2 //#hidden layers
#define TOTAL_LAYER 4 //#di layers

//Streams Settings
#define NSTREAMS 3

//Texture reference (FOR TARGET MATRIX)
texture<DATA, 2, cudaReadModeElementType> texreference_target;

//Constant memory (read by all the threads)
__constant__ DATA alpha_const[1];
__constant__ DATA eta_const[1];

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

/*Utility Struct for device -- useful for keeping together the parameters of Feedforward and Gradient Descent*/

typedef struct device_utility {
	DATA **WeightBiasDev;
	DATA **DeltaWeightBiasDev;
	DATA **H2HDev;
	DATA **DeltaDev;
	DATA **DeltaDevT; //DeltaDev trasposta
}util_dev;

/*deviceReduceBlockAtomicKernel*/
__inline__ __device__ DATA warpReduceSum(DATA);
__inline__ __device__ DATA blockReduceSum(DATA);
__global__ void deviceReduceBlockAtomicKernel(DATA *, DATA*, DATA, int);

/*MMMul(for feedforward)*/
__device__ void MMMulDevPartialFeed(DATA *, DATA *, DATA *, DATA *, DATA*, DATA *, DATA *, int, int, int, int, int, BOOL);
__global__ void MMMulDevFeed(DATA *, DATA *, DATA *, DATA *, DATA *, DATA*, DATA*, int, int, int, int, int, BOOL);
/*MMMul(for backpropagation)*/
__device__ void MMMulDevPartialBack(DATA *, DATA *, DATA *, DATA *, DATA *, int, int, int, int);
__global__ void MMMulDevBack(DATA *, DATA *, DATA *, DATA *, DATA *, int, int, int, int);
/*MMMul(for derivative)*/
__device__ void MMMulDevPartialDerivative(DATA *, DATA *, DATA *, int, int, int, int);
__global__ void MMMulDevDerivative(DATA *, DATA *, DATA *, int, int, int);
/*Apply Derivative*/
__global__ void updateWeightBiasMat(DATA *, DATA *, int, int);


/*HOST*/
void FeedAndBack(DATA *, util_dev *, DATA *, DATA *, int *, int, int, cudaStream_t *, BOOL);
void GradientDescent(util_dev *, int *, DATA, int, int, cudaStream_t *);

void HOST_feedforward(DATA *, DATA **, DATA **, DATA **, int *);
void printMat(DATA *, int, int);
void printErrorMat(DATA *, DATA*, int, int);
void MMMulHost(DATA *, DATA *, DATA *, DATA *, int, int, int);
BOOL matsAreEquals(DATA *, DATA *, int, int);
DATA errorReductionHost(DATA *, int, int);

/*HOST ALLOCATION AND INITIALIZATION AND CLEAN UP*/
void HOST_init_matrices(DATA **, DATA **, int*, int, DATA);
void HOST_clean(DATA **, DATA **, int);
/*DEVICE ALLOCATION AND CLEAN UP*/
void DEVICE_alloc_matrices(DATA**, DATA**, DATA**, DATA**, DATA**, int*, int, int);
void DEVICE_clean(DATA**, DATA**, DATA**, DATA**, DATA**, int);

/*----------------------------------------------------------------------MAIN---------------------------------------------------------------------------*/

int main(void) {

	DATA **WeightBias, **DeltaWeightBias;//Matrici utili all'host
	DATA **WeightBiasDev, **DeltaWeightBiasDev, **H2HDev, **DeltaDev, **DeltaDevT;//Matrici utili al device
	util_dev *ud = (util_dev*)malloc(sizeof(util_dev));

	DATA *INPUT_MAT, *ERROR_MAT, *DEV_ERROR_MAT;
	DATA *ERROR, *DEV_ERROR;
	DATA *TARGET;
	DATA alpha[1], eta[1];
	cudaStream_t streams[NSTREAMS];

	int *nupl = (int*)malloc(TOTAL_LAYER * sizeof(int));

	/*++++------------------------------------ERRORS--------------------------------------------------++++*/

	ERROR_MAT = (DATA*)malloc(TOTAL_PATT*NEURO_OUTPUT * sizeof(DATA)); // matrice di ERROR per il controllo di correttezza
	HANDLE_CUDA(cudaMalloc((void **)&DEV_ERROR_MAT, TOTAL_PATT*NEURO_OUTPUT * sizeof(DATA))); //matrice di ERROR del device

	ERROR = (DATA*)malloc(sizeof(DATA)); // ERROR per il controllo di correttezza
	HANDLE_CUDA(cudaMalloc((void **)&DEV_ERROR, sizeof(DATA))); // ERROR del device
	HANDLE_CUDA(cudaMemset(DEV_ERROR, 0, sizeof(DATA)));

	/*----------------------------------------ERRORS END--------------------------------------------------*/

	/*++++-----------------------------------init--alpha--and--eta--and--memcpy--to--symbol-----------------------------------++++*/

	//Momentum e learning rate presi dal file input.ml fornito dal prof
	alpha[0] = (DATA)2e-3;
	eta[0] = (DATA)2e-4;

	//il valore di default di cudaMemcpyKind è cudaMemcpyHostToDevice
	HANDLE_CUDA(cudaMemcpyToSymbol(alpha_const, alpha, sizeof(DATA)));
	HANDLE_CUDA(cudaMemcpyToSymbol(eta_const, eta, sizeof(DATA)));

	/*---------------------------------------end--init--alpha--and--eta--and--memcpy--to--symbol----------------------------------*/

	/*++++---------------------------init INPUT_MAT and TARGET (HOST)-----------------------------++++*/
	nupl[0] = NEURO_INPUT;
	nupl[1] = NEURO_H_0;
	nupl[2] = NEURO_H_1;
	nupl[TOTAL_LAYER - 1] = NEURO_OUTPUT;

	TARGET = (DATA*)malloc(NEURO_OUTPUT*TOTAL_PATT * sizeof(DATA)); //TARGET dei PATTERNS

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
			INPUT_MAT[i*NEURO_INPUT + j] = (DATA)rand() / (DATA)RAND_MAX;
		}
	}

	/*---------------------------end init INPUT_MAT and TARGET (HOST)-------------------------*/

	/*++++---------------------------Matrices--for--host--and--device-------------------------++++*/

	//Host 
	WeightBias = (DATA**)malloc((TOTAL_LAYER - 1) * sizeof(DATA*));
	DeltaWeightBias = (DATA**)malloc((TOTAL_LAYER - 1) * sizeof(DATA*));
	HOST_init_matrices(WeightBias, DeltaWeightBias, nupl, TOTAL_LAYER, alpha[0]);
	
	//Device
	WeightBiasDev = (DATA**)malloc((TOTAL_LAYER - 1) * sizeof(DATA*));
	DeltaWeightBiasDev = (DATA**)malloc((TOTAL_LAYER - 1) * sizeof(DATA*));
	H2HDev = (DATA**)malloc((TOTAL_LAYER) * sizeof(DATA*));
	DeltaDev = (DATA**)malloc((TOTAL_LAYER - 1) * sizeof(DATA*));
	DeltaDevT = (DATA**)malloc((TOTAL_LAYER - 1) * sizeof(DATA*));
	DEVICE_alloc_matrices(WeightBiasDev, DeltaWeightBiasDev, H2HDev, DeltaDev, DeltaDevT, nupl, TOTAL_LAYER, TOTAL_PATT);
	
	ud->WeightBiasDev = WeightBiasDev;
	ud->DeltaWeightBiasDev = DeltaWeightBiasDev;
	ud->H2HDev = H2HDev;
	ud->DeltaDev = DeltaDev;
	ud->DeltaDevT = DeltaDevT;

	/*---------------------------end--matrices--for--host--and--device----------------------------*/

	/*++++---------------------------cuda array for texture-----------------------------++++*/
	cudaArray* DEV_TARGET_CUDA;
	cudaChannelFormatDesc channel;

	channel = cudaCreateChannelDesc<DATA>();
	HANDLE_CUDA(cudaMallocArray(&DEV_TARGET_CUDA, &channel, NEURO_OUTPUT, TOTAL_PATT));
	HANDLE_CUDA(cudaMemcpyToArray(DEV_TARGET_CUDA, 0, 0, TARGET, NEURO_OUTPUT*TOTAL_PATT * sizeof(DATA), cudaMemcpyHostToDevice));

	texreference_target.filterMode = cudaFilterModePoint; //spegne l'interpolazione di cudaFilterModeLinear
	texreference_target.addressMode[0] = cudaAddressModeWrap;//funziona solo per coordinate normalizzate
	texreference_target.addressMode[1] = cudaAddressModeClamp;//funziona sia per coordinate normalizzate che non normalizzate

	HANDLE_CUDA(cudaBindTextureToArray(texreference_target, DEV_TARGET_CUDA)); //Binding della texture reference
	/*---------------------------end cuda array for texture-------------------------*/

	/*++++-----------Streams creation------------++++*/
	for (int i = 0; i < NSTREAMS; i++) {
		HANDLE_CUDA(cudaStreamCreate(&streams[i]));
	}
	/*---------------end--streams creation-----------*/

	/*++++-----------------------------------FEEDFORWARD--BACKPROPAGATION--AND--GRADIENT--DESCENT---------------------------------------++++*/

	cudaEvent_t start, stop;

	startTimer(&start, &stop);

	//Solo prima della prima epoca:
	for (int i = 0; i < (TOTAL_LAYER - 1); i++) {
		HANDLE_CUDA(cudaMemcpy(WeightBiasDev[i], WeightBias[i], (nupl[i] + 1)*nupl[i + 1] * sizeof(DATA), cudaMemcpyHostToDevice));
		HANDLE_CUDA(cudaMemcpy(DeltaWeightBiasDev[i], DeltaWeightBias[i], (nupl[i] + 1)*nupl[i + 1] * sizeof(DATA), cudaMemcpyHostToDevice));
	}

	FeedAndBack(INPUT_MAT, ud, DEV_ERROR_MAT, DEV_ERROR, nupl, TOTAL_LAYER, TOTAL_PATT, streams, 1);
	//cudaDeviceSynchronize();//
	GradientDescent(ud, nupl, eta[0], TOTAL_LAYER, TOTAL_PATT, streams);
	stopAndPrint(&start, &stop);

	HANDLE_CUDA(cudaMemcpy(ERROR, DEV_ERROR, sizeof(DATA), cudaMemcpyDeviceToHost));
	printf("Reduced Error for epoch 0: %f\n\n", *ERROR);

	for (int epoch = 1; epoch < MAX_EPOCHS; epoch++) {

		HANDLE_CUDA(cudaMemset(DEV_ERROR, 0, sizeof(DATA)));//Necessario per 'ripulire' dev_error del valore dell'epoca precedente

		startTimer(&start, &stop);
		FeedAndBack(INPUT_MAT, ud, DEV_ERROR_MAT, DEV_ERROR, nupl, TOTAL_LAYER, TOTAL_PATT, streams, 0);
		//cudaDeviceSynchronize();//
		GradientDescent(ud, nupl, eta[0], TOTAL_LAYER, TOTAL_PATT, streams);
		stopAndPrint(&start, &stop);

		HANDLE_CUDA(cudaMemcpy(ERROR, DEV_ERROR, sizeof(DATA), cudaMemcpyDeviceToHost));//La cudaMemcpy in questo caso è SINCRONA e quindi funge da punto
		printf("Reduced Error for epoch %d: %f\n\n", epoch, *ERROR);					// di sincronizzazione
	}
	
	/*
	HANDLE_CUDA(cudaMemcpy(ERROR_MAT, DEV_ERROR_MAT, TOTAL_PATT*NEURO_OUTPUT * sizeof(DATA), cudaMemcpyDeviceToHost));
	//printMat(ERROR_MAT, TOTAL_PATT, NEURO_OUTPUT);
	DATA red_host = errorReductionHost(ERROR_MAT, TOTAL_PATT, NEURO_OUTPUT);
	printf("host reduction error : %f\n", red_host);
	*/

	/*---------------------------------------END--FEEDFORWARD--BACKPROPAGATION--AND--GRADIENT--DESCENT--------------------------------------*/

	/*++++--------------------------------deallocations------------------------------------++++*/
	//Deallocations
	HOST_clean(WeightBias, DeltaWeightBias, TOTAL_LAYER);
	free(nupl);
	free(TARGET);
	free(ERROR_MAT);
	free(ud);
	DEVICE_clean(WeightBiasDev, DeltaWeightBiasDev, H2HDev, DeltaDev, DeltaDevT, TOTAL_LAYER);
	cudaFree(DEV_ERROR_MAT);
	cudaFree(DEV_ERROR);
	cudaFreeHost(INPUT_MAT);
	//Unbinding texture
	cudaUnbindTexture(texreference_target);
	//Free cuda array
	cudaFreeArray(DEV_TARGET_CUDA);
	/*++++-----------Removing streams------------++++*/
	for (int i = 0; i < NSTREAMS; i++) {
		HANDLE_CUDA(cudaStreamDestroy(streams[i]));
	}
	/*---------------end--Removing streams-----------*/
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

/*
Nella chiamata di questo kernel è meglio usare una griglia lineare di 8 blocchi con 256 threads ciascuno --
In tal modo vengono limitati gli accessi alla shared memory rispetto all'implementazione con 2 blocchi da 1024 threads ciascuno
Attenzione ai possibili arrotondamenti di numeri a virgola mobile dovuti alle atomicAdd.
*/
__global__ void deviceReduceBlockAtomicKernel(DATA *in, DATA* out, DATA eta, int N) {
	DATA sum = 0.0f;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
		sum += eta*in[i];
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) 
		atomicAdd(out, sum);
}

/*-------------------------------end--deviceReduceBlockAtomicKernel--------------------------*/

/*++++---------------------------MMMul--Feedforward-------------------------++++*/

/*
h2h è il puntatore alla porzione dell'h2h globale da considerare in questa fase
(ad ogni passo il kernel che invoca questo device incrementa il puntatore h2h
in modo proporzionale al patt_per_step (e similmente h2h_dest) (vedi sotto)).
offset_y è la posizione considerata lungo le y (nelle matrici h2h, h2h_dest ed eventualmente error) durante la chiamata corrente a questo __device__.
DeltaO è calcolato per l'output layer (out_layer == 1) e propagato poi con backpropagation.
Anche error è calcolato per l'output layer (out_layer == 1) ma non viene propagato all'indietro.
*/

__device__ void MMMulDevPartialFeed(DATA *h2h, DATA *w, DATA *biases, DATA *h2h_dest, DATA *delta, DATA *deltaT, DATA *error, int row_w, int col_w, int current_patt, int offset_y, int tot_patt, BOOL out_layer) {

	int tx = threadIdx.x, ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int block_dim = blockDim.x; // assumiamo che i blocchi siano quadrati
	int dest_x = block_x*block_dim + tx;
	int dest_y = block_y*block_dim + ty;

	int w_x = block_x*block_dim; // start block in w
	int h2h_y = block_y*block_dim*row_w; // start block in h2h

	int end_h2h = h2h_y + row_w - 1; // posizione di last block in h2h

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
		shared_h2h[ty][tx] = (t_index_h2h < current_patt*row_w) ? (h2h[t_index_h2h]) : (0.0f);
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
	if (dest_x < col_w && dest_y < current_patt) {

		DATA out = (DATA)1.0 / (DATA)(1.0 + exp(-(partial + biases[dest_x])));
		h2h_dest[dest_y*col_w + dest_x] = out; //SIGMA

		//Se siamo nell'ultimo passo
		if (out_layer) {

			DATA target = tex2D(texreference_target, dest_x, dest_y + offset_y);

			//Scrivi nella posizione corrispondente della matrice di ERRORE
			/*0.5*(Target[p][k] - Output[p][k])*(Target[p][k] - Output[p][k])*/
			error[dest_y*col_w + dest_x] = 0.5*(target - out)*(target - out);

			//Scrivi nella posizione corrispondente della matrice DELTA
			/*(Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k])*/
			delta[dest_y*col_w + dest_x] = (target - out)*(out)*(1 - out);
			//Versione trasposta della matrice DELTA
			deltaT[dest_x*tot_patt + dest_y] = (target - out)*(out)*(1 - out);
		}
	}
}


/*
Questo kernel ad ogni passo incrementa il puntatore ad h2h di patt_per_step*row_w (e similmente h2h_dest di patt_per_step*col_w).
Calcola poi i current_patts come MIN(remaining_patts, patt_per_step), i quali sono i pattern che devono essere considerati nella
prossima chiamata al device MMMulDevPartialFeed.
Il controllo if (pos_block_y >= current_patts) occorre per terminare tutti quei blocchi della griglia di questo kernel che sono fuori
dal range dei pattern da considerare nel __device__ MMMulDevPartialFeed.
Bisogna ricordare che ogni stream che invoca questo kernel visiona una porzione dei pattern che è pari a stream_size, salvo l'aggiunta
del possibile resto della divisione total_pattern/NSTREAMS.
*/
__global__ void MMMulDevFeed(DATA *h2h, DATA *w, DATA *biases, DATA *h2h_dest, DATA *delta, DATA *deltaT, DATA *error, int row_w, int col_w, int stream_offset_y, int stream_size, int tot_patt, BOOL out_layer) {

	int current_patts;
	int remaining_patts;
	int patt_per_step = gridDim.y*blockDim.y; //è il numero di pattern (quando possibile...) da considerare in ciascuna iterazione su h2h e quindi su h2h_dest
	int pos_block_y = blockIdx.y*blockDim.x;  //Posizione del blocco corrente rispetto alla griglia lungo le y
											  //Assumiamo che i blocchi siano quadrati (blockDim.x = blockDim.y)		

	for (int y = 0; y < stream_size; y += patt_per_step) {

		remaining_patts = stream_size - y;
		current_patts = MIN(remaining_patts, patt_per_step);

		if (pos_block_y >= current_patts) { return; }

		MMMulDevPartialFeed(h2h + y*row_w, w, biases, h2h_dest + y*col_w, delta + y*NEURO_OUTPUT, deltaT + y, error + y*NEURO_OUTPUT, row_w, col_w, current_patts, stream_offset_y + y, tot_patt, out_layer);
	}
}

/*-------------------------------end--MMMul--Feedforward------------------------*/

/*++++---------------------------MMMul--BackPropagation-------------------------++++*/

__device__ void MMMulDevPartialBack(DATA *delta_l, DATA *w, DATA *delta_l_1, DATA *delta_l_1T, DATA *h2h_l_1, int row_w, int col_w, int num_pattern, int total_patt) {

	int tx = threadIdx.x, ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int block_dim = blockDim.x; // assumiamo che i blocchi siano quadrati
	int dest_x = block_x*block_dim + tx;
	int dest_y = block_y*block_dim + ty;

	//Dobbiamo scorrere la matrice w per righe (stiamo considerando w come fosse trasposta -- vedi 16 febbraio su diario)
	int w_y = block_x*block_dim*col_w; // start block in w
	int delta_l_y = block_y*block_dim*col_w; // start block in delta_l_y

	int end_delta_l = delta_l_y + col_w - 1; // posizione di last block in delta_l

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
		//Salviamo la sottomatrice trasposta della matrice dei pesi nella shared memory (osservare che in tal modo evitiamo conflitti di banco):
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
		//Propaga indietro il delta al layer precedente
		DATA h2h_l_1_target = h2h_l_1[dest_y*row_w + dest_x];
		delta_l_1[dest_y*row_w + dest_x] = partial*h2h_l_1_target*(1 - h2h_l_1_target);
		delta_l_1T[dest_x*total_patt + dest_y] = partial*h2h_l_1_target*(1 - h2h_l_1_target);
	}
}

__global__ void MMMulDevBack(DATA *delta_l, DATA *w, DATA *delta_l_1, DATA *delta_l_1T, DATA *h2h_l_1, int row_w, int col_w, int stream_size, int tot_patt) {

	int current_patts;
	int remaining_patts;
	int patt_per_step = gridDim.y*blockDim.y;
	int pos_block_y = blockIdx.y*blockDim.x; //Posizione del blocco corrente rispetto alla griglia lungo le y
											 //Assumiamo che i blocchi siano quadrati (blockDim.x = blockDim.y)		

	for (int y = 0; y < stream_size; y += patt_per_step) {

		remaining_patts = stream_size - y;
		current_patts = MIN(remaining_patts, patt_per_step);

		if (pos_block_y >= current_patts) { return; }

		MMMulDevPartialBack(delta_l + y*col_w, w, delta_l_1 + y*row_w, delta_l_1T + y, h2h_l_1 + y*row_w, row_w, col_w, current_patts, tot_patt);
	}
}

/*-------------------------------end--MMMul--BackPropagation------------------------*/

/*-------------------------------Compute--Derivative----------------------------------*/

/*
current_rows sono il numero di righe attualmente considerate nella matrice dev_c e, quindi, il numero di righe attualmente considerate nella trasposta della
matrice dev_a
*/
__device__ void MMMulDevPartialDerivative(DATA *dev_a, DATA *dev_b, DATA *dev_c, int col_dev_a, int col_dev_b, int current_rows, int patterns) {

	int tx = threadIdx.x, ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int block_dim = blockDim.x; // assumiamo che i blocchi siano quadrati
	int dest_x = block_x*block_dim + tx;
	int dest_y = block_y*block_dim + ty;

	int dev_a_x = block_y*block_dim;
	int dev_b_x = block_x*block_dim;

	int end_dev_a = dev_a_x + col_dev_a*(patterns - 1);

	int step_dev_a = col_dev_a*block_dim;
	int step_dev_b = col_dev_b*block_dim;

	DATA partial = 0.0f;
	int block_r_border = 0; // contatore che indica in che iterazione dei blocchi ci troviamo
	int current_inc;
	int min;

	for (int dev_a_id = dev_a_x, dev_b_id = dev_b_x; dev_a_id <= end_dev_a; dev_a_id += step_dev_a, dev_b_id += step_dev_b) {

		block_r_border += block_dim;

		__shared__ DATA shared_dev_a[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER];
		__shared__ DATA shared_dev_b[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER];

		int t_index_dev_a = dev_a_id + tx + ty*col_dev_a;
		int t_index_dev_b = dev_b_id + tx + ty*col_dev_b;

		//Salviamo la sottomatrice trasposta della matrice dev_a nella shared memory (osservare che in tal modo evitiamo conflitti di banco):
		shared_dev_a[tx][ty] = (t_index_dev_a < patterns*col_dev_a) ? (dev_a[t_index_dev_a]) : (0.0f);
		shared_dev_b[ty][tx] = (t_index_dev_b < patterns*col_dev_b) ? (dev_b[t_index_dev_b]) : (0.0f);

		__syncthreads();

		current_inc = patterns - (block_r_border - block_dim);
		min = MIN(current_inc, block_dim);

		#pragma unroll(2)
		for (int k = 0; k < min; k++) {
			partial += eta_const[0] * shared_dev_a[ty][k] * shared_dev_b[k][tx];
		}

		__syncthreads();
	}

	if (dest_x < col_dev_b && dest_y < current_rows) {
		atomicAdd(&dev_c[dest_y*col_dev_b + dest_x], partial);
	}
}

__global__ void MMMulDevDerivative(DATA *h2h, DATA *delta, DATA *delta_weight, int col_h2h, int col_delta, int patterns) {

	int current_rows;
	int remaining_rows;
	int pos_block_y = blockIdx.y*blockDim.x; //Posizione del blocco corrente rispetto alla griglia lungo le y
											 //Assumiamo che i blocchi siano quadrati (blockDim.x = blockDim.y)		
	int rows_per_step = gridDim.y*blockDim.y; //Righe della matrice delle derivate dei pesi da considerare (al massimo) ogni step

	for (int y = 0; y < col_h2h; y += rows_per_step) {

		remaining_rows = col_h2h - y;
		current_rows = MIN(remaining_rows, rows_per_step);

		if (pos_block_y >= current_rows) { return; }
		MMMulDevPartialDerivative(h2h + y, delta, delta_weight + y*col_delta, col_h2h, col_delta, current_rows, patterns);
	}
}

/*-------------------------------End--Compute--Derivative-----------------------------*/

/*-------------------------------Apply--Derivative------------------------------------*/

__global__ void updateWeightBiasMat(DATA *delta_weightbias, DATA *weight, int rows, int cols) {

	int dest_x = blockIdx.x*blockDim.x + threadIdx.x;
	int dest_y = blockIdx.y*blockDim.y + threadIdx.y;

	if (dest_x < cols && dest_y < rows) {
		DATA derivative = delta_weightbias[dest_y*cols + dest_x];
		weight[dest_y*cols + dest_x] += derivative;
		delta_weightbias[dest_y*cols + dest_x] *= alpha_const[0];
	}
}

/*--------------------------End--Apply--Derivative------------------------------------*/

/*HOST*/

/*FASI DI FEEDFORWARD E BACKPROPAGATION -- L'INPUT VIENE PROPAGATO ATTRAVERSO LA RETE E POI DI NUOVO ALL'INDIETRO*/
void FeedAndBack(DATA *INPUT, util_dev *ud, DATA *dev_error_mat, DATA *dev_error, int *nupl, int layers, int patterns, cudaStream_t *streams, BOOL first_epoch) {
	//cudaEvent_t start, stop;

	//Stream size e stream remainder
	int stream_size = patterns / NSTREAMS;
	int stream_remainder = patterns % NSTREAMS;
	//Setting della Grid
	dim3 grid, block;
	//int patt_per_step;
	//Puntatori utili
	DATA *h2h, *w, *bias, *h2h_dest, *delta, *deltaT, *error;
	//Delta da cui parte l'informazione (delta_l) e delta dove arriva tramite la backpropagation (delta_l_1)
	DATA *delta_l, *delta_l_1, *delta_l_1T;

	//offset
	int offset;

	for (int i = 0; i < NSTREAMS; i++) {

		//Leggere 15 febbraio del diario (passo 1 del feedforward, considerazioni)
		block.x = gs.block[0];
		block.y = gs.block[0];
		grid.x = (nupl[1] + block.x - 1) / block.x;
		grid.y = MAX(gs.grid[0] / grid.x, 1); //Evitare che possa diventare 0

		//patt_per_step = grid.y * block.y;
		//Lasciare in questa posizione
		offset = i*stream_size;
		//Per l'ultimo stream va aggiunto il resto dei pattern (non è detto che patterns sia divisibile per 3)
		if (i == (NSTREAMS - 1)) { stream_size += stream_remainder; }

		//Settare i puntatori
		h2h = (ud->H2HDev)[0] + offset*nupl[0];
		w = (ud->WeightBiasDev)[0] + nupl[1];
		bias = (ud->WeightBiasDev)[0];
		h2h_dest = (ud->H2HDev)[1] + offset*nupl[1];
		delta = (ud->DeltaDev)[layers - 2] + offset*nupl[layers - 1];
		deltaT = (ud->DeltaDevT)[layers - 2] + offset; //Basta offset perché è la versione trasposta di delta
		error = dev_error_mat + offset*nupl[layers - 1];
		//Puntatori settati

		if (first_epoch) {
			HANDLE_CUDA(cudaMemcpyAsync(h2h, INPUT + offset*nupl[0], nupl[0] * stream_size * sizeof(DATA), cudaMemcpyHostToDevice, streams[i]));
		}

		//Primo step del feedforward:
		MMMulDevFeed << <grid, block, 0, streams[i] >> > (h2h, w, bias, h2h_dest, delta, deltaT, error, nupl[0], nupl[1], offset, stream_size, patterns, 0);

		for (int l = 1; l < (layers - 2); l++) {

			block.x = gs.block[l];
			block.y = gs.block[l];
			grid.x = (nupl[l + 1] + block.x - 1) / block.x;
			grid.y = MAX(gs.grid[l] / grid.x, 1); //Evitare che possa diventare 0

			//patt_per_step = grid.y * block.y;
			//Settare i puntatori
			h2h = (ud->H2HDev)[l] + offset*nupl[l];
			w = (ud->WeightBiasDev)[l] + nupl[l+1];
			bias = (ud->WeightBiasDev)[l];
			h2h_dest = (ud->H2HDev)[l + 1] + offset*nupl[l + 1];
			//Delta e error già settati
			//Puntatori settati

			MMMulDevFeed << <grid, block, 0, streams[i] >> > (h2h, w, bias, h2h_dest, delta, deltaT, error, nupl[l], nupl[l + 1], offset, stream_size, patterns, 0);
		}

		//Ultimo step del Feedforward:

		block.x = gs.block[layers - 2];
		block.y = gs.block[layers - 2];
		grid.x = (nupl[layers - 1] + block.x - 1) / block.x;
		grid.y = MAX(gs.grid[layers - 2] / grid.x, 1); //Evitare che possa diventare 0

		//patt_per_step = grid.y * block.y;
		//Settare i puntatori
		h2h = (ud->H2HDev)[layers - 2] + offset*nupl[layers - 2];
		w = (ud->WeightBiasDev)[layers - 2] + nupl[layers - 1];
		bias = (ud->WeightBiasDev)[layers - 2];
		h2h_dest = (ud->H2HDev)[layers - 1] + offset*nupl[layers - 1];
		//Delta e error già settati
		//Puntatori settati

		MMMulDevFeed << <grid, block, 0, streams[i] >> > (h2h, w, bias, h2h_dest, delta, deltaT, error, nupl[layers - 2], nupl[layers - 1], offset, stream_size, patterns, 1);

		//BackPropagation per tutti gli streams iterando tra i layers:
		for (int delta_index = (layers - 2); delta_index > 0; delta_index--) {

			block.x = gs.block[delta_index];
			block.y = gs.block[delta_index];
			grid.x = (nupl[delta_index] + block.x - 1) / block.x;
			grid.y = MAX(gs.grid[delta_index] / grid.x, 1); //Evitare che possa diventare 0

			//patt_per_step = grid.y * block.y;
			//Settare i puntatori
			h2h = (ud->H2HDev)[delta_index] + offset*nupl[delta_index];
			w = (ud->WeightBiasDev)[delta_index] + nupl[delta_index + 1];
			delta_l = (ud->DeltaDev)[delta_index] + offset*nupl[delta_index + 1];
			delta_l_1 = (ud->DeltaDev)[delta_index - 1] + offset*nupl[delta_index];
			delta_l_1T = (ud->DeltaDevT)[delta_index - 1] + offset; //Basta offset perché è la versione trasposta di delta_l_1
			//Puntatori settati
			MMMulDevBack << <grid, block, 0, streams[i] >> > (delta_l, w, delta_l_1, delta_l_1T, h2h, nupl[delta_index], nupl[delta_index + 1], stream_size, patterns);
		}
	}
	//Riduzione dell'errore (default stream)
	deviceReduceBlockAtomicKernel << <OPTIMUM_BLOCK_NUM * 2, BLOCK_SIDE*BLOCK_SIDE >> > (dev_error_mat, dev_error, 1.0f, patterns*nupl[layers - 1]);
}

/*
FASE DI DISCESA DEL GRADIENTE -- VENGONO MODIFICATE LE MATRICI DELTAWEIGHT E DELTABIAS E VENGONO AGGIORNATI I VALORI DEI PESI E DEI BIAS CON LE RISPETTIVE DERIVATE
N.B. PER ORA AGGIORNIAMO SOLO LE MATRICI DEI DELTAWEIGHT E I VALORI DEI PESI. BISOGNERA' AGGIORNARE ANCHE I DELTABIAS E I VALORI DEI BIAS...
*/
void GradientDescent(util_dev *ud, int *nupl, DATA eta, int layers, int patterns, cudaStream_t *streams) {
	
	//Puntatori utili
	DATA *h2h, *delta, *deltaT, *delta_weight, *delta_bias, *weight;

	//Stream size e stream remainder
	int stream_size = patterns / NSTREAMS;
	int stream_remainder = patterns % NSTREAMS;
	//Setting della Grid
	dim3 grid, block;

	int offset;

	for (int s = 0; s < NSTREAMS; s++) {

		//Lasciare in questa posizione
		offset = s*stream_size;
		//Per l'ultimo stream va aggiunto il resto dei pattern (non è detto che patterns sia divisibile per 3)
		if (s == (NSTREAMS - 1)) { stream_size += stream_remainder; }

		for (int delta_index = (layers - 2); delta_index >= 0; delta_index--) {
			//Ricordare che la griglia deve 'adagiarsi' sulla matrice di output che adesso è delta_weight che ha dimensioni nupl[delta_index]
			//lungo le y e nupl[delta_index + 1] lungo le x.
			block.x = gs.block[delta_index];
			block.y = gs.block[delta_index];
			grid.x = (nupl[delta_index + 1] + block.x - 1) / block.x;
			grid.y = MAX(gs.grid[delta_index] / grid.x, 1); //Evitare che possa diventare 0

			//Settare i puntatori
			h2h = (ud->H2HDev)[delta_index] + offset*nupl[delta_index];
			delta = (ud->DeltaDev)[delta_index] + offset*nupl[delta_index + 1];
			delta_weight = (ud->DeltaWeightBiasDev)[delta_index] + nupl[delta_index + 1];//Non considero la prima riga dei delta_bias in (ud->DeltaWeightBiasDev)[delta_index]
			//Puntatori settati

			//Calcolare le derivate dei pesi
			MMMulDevDerivative << <grid, block, 0, streams[s] >> > (h2h, delta, delta_weight, nupl[delta_index], nupl[delta_index + 1], stream_size);

			//Settare i puntatori
			deltaT = (ud->DeltaDevT)[delta_index];
			delta_bias = (ud->DeltaWeightBiasDev)[delta_index];//La prima riga di (ud->DeltaWeightBiasDev)[delta_index] è la riga dei bias
			//Puntatori settati

			//Calcolare le derivate dei bias del layer (delta_index + 1), scorrendo i neuroni del layer (delta_index + 1)
			for (int k = s; k < nupl[delta_index + 1]; k += NSTREAMS) {
				deviceReduceBlockAtomicKernel << <OPTIMUM_BLOCK_NUM, BLOCK_SIDE*BLOCK_SIDE, 0, streams[s] >> > (deltaT + k*patterns, delta_bias + k, eta, patterns);
			}
		}
	}
	
	//Default stream (sincronizza gli altri stream--guardare la documentazione):
	for (int l = 0; l < (layers - 1); l++) {

		block.x = gs.block[l];
		block.y = gs.block[l];
		grid.x = (nupl[l + 1] + block.x - 1) / block.x;
		grid.y = ((nupl[l] + 1) + block.y - 1) / block.y;

		//Settare i puntatori
		delta_weight = (ud->DeltaWeightBiasDev)[l];//In questo caso prendiamo le derivate sia dei pesi che dei bias del livello corrente
		weight = (ud->WeightBiasDev)[l];//Applichiamo le derivate di cui sopra sia ai pesi che ai bias del livello corrente
		//Puntatori settati

		updateWeightBiasMat << <grid, block >> > (delta_weight, weight, nupl[l] + 1, nupl[l + 1]);
	}
}

/*UTILITY FUNCTIONS*/

void HOST_feedforward(DATA *INPUT, DATA **W, DATA **BIAS, DATA **H2H, int *nupl) {

	MMMulHost(INPUT, W[0], BIAS[0], H2H[0], TOTAL_PATT, nupl[0], nupl[1]);
	MMMulHost(H2H[0], W[1], BIAS[1], H2H[1], TOTAL_PATT, nupl[1], nupl[2]);
	MMMulHost(H2H[1], W[2], BIAS[2], H2H[2], TOTAL_PATT, nupl[2], nupl[3]);
}

/*Stampa una matrice*/
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

/*Stampa la matrice di errore sull'host (per il controllo di correttezza del device)*/
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

/*Moltiplicazione sull'host*/
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

/*Controllo di correttezza del device*/
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

/*Controlla la riduzione del device*/
DATA errorReductionHost(DATA *error_mat, int rows, int cols) {

	DATA reduction = 0.0f;

	for (int i = 0; i < rows*cols; i++) {
		reduction += error_mat[i];
	}

	return reduction;
}

/*ALLOCATION AND CLEAN UP FUNCTIONS*/

/*Alloca e inizializza le matrici sull'host*/
void HOST_init_matrices(DATA **WeightBias, DATA **DeltaWeightBias, int* nupl, int layers, DATA alpha) {
	
	for (int i = 0; i < (layers - 1); i++) {

		WeightBias[i] = (DATA*)malloc((nupl[i] + 1)*nupl[i + 1] * sizeof(DATA));
		DeltaWeightBias[i] = (DATA*)malloc((nupl[i] + 1)*nupl[i + 1] * sizeof(DATA));

		for (int j = 0; j < (nupl[i] + 1); j++) {
			for (int k = 0; k < nupl[i + 1]; k++) {
				WeightBias[i][j*nupl[i + 1] + k] = (DATA)rand() / (DATA)RAND_MAX;
				DeltaWeightBias[i][j*nupl[i + 1] + k] = alpha*0.0f;
			}
		}
	}
}

/*Rimuove le risorse dell'host dallo heap*/
void HOST_clean(DATA **WeightBias, DATA **DeltaWeightBias, int layers) {

	for (int i = 0; i < (layers - 1); i++) {
		free(WeightBias[i]);
		free(DeltaWeightBias[i]);
	}
	free(WeightBias);
	free(DeltaWeightBias);
}

/*Alloca le matrici sul device*/
void DEVICE_alloc_matrices(DATA **WeightBiasDev, DATA **DeltaWeightBiasDev, DATA **H2HDev, DATA **DeltaDev, DATA **DeltaDevT, int *nupl, int layers, int patterns) {

	for (int i = 0; i < (layers - 1); i++) {
		HANDLE_CUDA(cudaMalloc(&(WeightBiasDev[i]), (nupl[i] + 1)*nupl[i + 1] * sizeof(DATA)));
		HANDLE_CUDA(cudaMalloc(&(DeltaWeightBiasDev[i]), (nupl[i] + 1)*nupl[i + 1] * sizeof(DATA)));

		HANDLE_CUDA(cudaMalloc(&(H2HDev[i]), patterns*nupl[i]*sizeof(DATA)));
		HANDLE_CUDA(cudaMalloc(&(DeltaDev[i]), patterns*nupl[i + 1] * sizeof(DATA)));
		HANDLE_CUDA(cudaMalloc(&(DeltaDevT[i]), nupl[i + 1] * patterns * sizeof(DATA)));
	}
	HANDLE_CUDA(cudaMalloc(&(H2HDev[layers - 1]), patterns*nupl[layers - 1] * sizeof(DATA)));
}

/*Rimuove le risorse del device dallo heap*/
void DEVICE_clean(DATA **WeightBiasDev, DATA **DeltaWeightBiasDev, DATA **H2HDev, DATA **DeltaDev, DATA **DeltaDevT, int layers) {
	
	for (int i = 0; i < (layers - 1); i++) {
		HANDLE_CUDA(cudaFree(WeightBiasDev[i]));
		HANDLE_CUDA(cudaFree(DeltaWeightBiasDev[i]));

		HANDLE_CUDA(cudaFree(H2HDev[i]));
		HANDLE_CUDA(cudaFree(DeltaDev[i]));
		HANDLE_CUDA(cudaFree(DeltaDevT[i]));
	}
	HANDLE_CUDA(cudaFree(H2HDev[layers - 1]));

	free(WeightBiasDev);
	free(DeltaWeightBiasDev);
	free(H2HDev);
	free(DeltaDev);
	free(DeltaDevT);
}