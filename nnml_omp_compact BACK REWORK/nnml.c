/*******************************************************************************
*   original code by                                                           *
*   JOHN BULLINARIA  2004. Modified by Massimo Bernaschi 2016                  *
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h> // OpenMP

#include "common.h"
#include "dictionary.h"
#include "iniparser.h"

#define REAL float
#define NULLFILE "/dev/null"
#define DEFMAXEPOCH 1000
#if !defined(MAX)
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif

void Usage(char *cmd)
{
    printf("-----------------------\n");
    printf("Neural Networks Learning Code (by backpropagation)\n");
    printf("-----------------------\n");
    printf("Usage: %s \n"
           "-i inputfile  \n"
           "[-v verbose \n -D Debug \n  -h ThisHelp]\n",
           cmd);
}

/**Popola la matrice con nrow e ncol. Il tipo di dato degli elementi della matrice dipende dal parametro ts (esprime la sizeof)**/
void ReadFromFile(char *fn,void **array,int nrow, int ncol, int ts)
{
    FILE *fp=NULL;
    int i, j;
    double **dp;
    float  **sp;
    switch(ts)
    {
    case sizeof(float):
        sp=(float **)array;
        break;
    case sizeof(double):
        dp=(double **)array;
        break;
    default:
        writelog(TRUE,APPLICATION_RC,"invalid size in ReadFromFile: %d\n",ts);
        break;
    }
    fp=Fopen(fn,"r");
    for(i=0; i<nrow; i++)
    {
        for(j=0; j<ncol; j++)
        {
            switch(ts)
            {
            case sizeof(float):
                fscanf(fp,"%f",&(sp[i][j]));
                break;
            case sizeof(double):
                fscanf(fp,"%lf",&(dp[i][j]));
                break;
            default:
                writelog(TRUE,APPLICATION_RC,"invalid size in ReadFromFile: %d\n",ts);
                break;
            }
        }
    }
    fclose(fp);
}


int main(int argc, char *argv[])
{
    int     h, i, j, k, p, np, op, epoch;
    int    NumPattern, NumInput, NumHidden, NumOutput;
    int nthreads; // For openMP
    double startime; //For openMP
    int j_delta, i_delta, k_delta; // for Delta calculations (For openMP)
    /*
        double Input[NUMPAT+1][NUMIN+1] = { 0, 0, 0,  0, 0, 0,  0, 1, 0,  0, 0, 1,  0, 1, 1 };
        double Target[NUMPAT+1][NUMOUT+1] = { 0, 0,  0, 0,  0, 1,  0, 1,  0, 0 };
        double SumH[NUMPAT+1][NUMHID+1], WeightIH[NUMIN+1][NUMHID+1], Hidden[NUMPAT+1][NUMHID+1];
        double SumO[NUMPAT+1][NUMOUT+1], WeightHO[NUMHID+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];
        double DeltaO[NUMPAT+1][NUMOUT+1], SumDOW[NUMHID+1], DeltaH[NUMPAT+1][NUMHID+1];
        double DeltaWeightIH[NUMIN+1][NUMHID+1], DeltaWeightHO[NUMHID+1][NUMOUT+1];
        double Error, eta = 0.5, alpha = 0.9, smallwt = 0.5;
    */
    REAL **Input;
    REAL **Target;
    REAL **Sum;
    REAL **WeightIH, **Hidden;
    REAL ***H2H, ***DeltaH2H;
    REAL ***WeightH2H, ***DeltaWeightH2H;
    REAL **Htemp, **Deltatemp;
    REAL **WeightHO, **Output;
    REAL **DeltaO, *SumDOW, **DeltaH;
    REAL **DeltaWeightIH, **DeltaWeightHO;

    REAL *threadError; // OpenMP -- cumulative error per/thread
    REAL ***threadDeltaHO; // OpenMP -- cumulative Delta for weights and biases (i,j) for output layer
    REAL ***threadDeltaH2H; // OpenMP -- cumulative Delta for weights and biases (i,j) for hidden layers

    REAL Error, Eps, eta, alpha, smallwt;
    int *ranpat;
    int verbose=FALSE;
    int maxepoch=DEFMAXEPOCH;
    int dimsum=0;
    int NumHL=1;
    int *nupl=NULL;
    char *inputfile = NULL;
    char *po;
    dictionary *ini;
    char key[MAXSTRLEN];
    char formatstring[MAXSTRLEN];
    char LogFileName[MAXSTRLEN];
    char InputFileName[MAXSTRLEN];
    char TargetFileName[MAXSTRLEN];
    char ResultFileName[MAXSTRLEN];
    char RestartFileName[MAXSTRLEN];
    char DeltaFileName[MAXSTRLEN];
    char RestartDeltaFileName[MAXSTRLEN];
    FILE *fp=NULL;
    FILE *fpd=NULL;
    FILE *fpl=NULL;
    if(sizeof(REAL)==sizeof(float))
    {
        strcpy(formatstring,"%f ");
    }
    if(sizeof(REAL)==sizeof(double))
    {
        strcpy(formatstring,"%lf ");
    }

    for(i = 1; i < argc; i++)
    {
        po = argv[i];
        if (*po++ == '-')
        {
            switch (*po++)
            {
            case 'h':
                Usage(argv[0]);
                exit(OK);
                break;
            case 'v':
                verbose=TRUE;
                break;
            case 'i':
                SKIPBLANK
                inputfile=Strdup(po);
                break;
            default:
                Usage(argv[0]);
                exit(OK);
                break;
            }
        }
    }
    if(inputfile==NULL)
    {
        Usage(argv[0]);
        exit(OK);
    }

    ini = iniparser_load(inputfile); // file of configuration (hidden layers, learning rate, max number of epochs ...)

    if(ini==NULL)
    {
        writelog(TRUE,APPLICATION_RC,"Cannot parse file: %s\n", inputfile);
    }

    READINTFI(maxepoch,"Max number of epochs");
    READINTFI(nthreads,"Number of threads"); // NUMBER OF THREADS FOR PARALLELIZATION
    READINTFI(NumPattern,"Number of training data");
    READINTFI(NumInput,"Number of input units");
    READINTFI(NumOutput,"Number of output units");
    READINTFI(NumHL,"Number of hidden layers");
    READREALFI(eta,"Learning rate");
    READREALFI(alpha,"Momentum");
    READREALFI(smallwt,"Initialization scale");
    READREALFI(Eps,"Error threshold");
    {
        READSTRFI(LogFileName,"Log file name");
    }
    {
        READSTRFI(InputFileName,"Input file name");
    }
    {
        READSTRFI(TargetFileName,"Target file name");
    }
    {
        READSTRFI(ResultFileName,"Results file name");
    }
    {
        READSTRFI(DeltaFileName,"Result delta file name");
    }
    {
        READSTRFI(RestartFileName,"Restart file name");
    }
    {
        READSTRFI(RestartDeltaFileName,"Restart delta file name");
    }
    nupl=makevect(NumHL+2,sizeof(int)); // vector of (num hidden layer + 2) elements: vec[i] -> # of sigmoid neurons of layer i
    nupl[0]=NumInput; // Num of sigmoid neurons in input layer of this neural network
    nupl[NumHL+1]=NumOutput; // Num of sigmoid neurons in output layer of this neural network

    if(NumHL)   // If there are hidden layers
    {
        int scratch;
        char tempstring[MAXSTRLEN];
        H2H=(REAL ***)Malloc(sizeof(REAL *)*(NumHL)); // Pointers to matrices (like vector of matrices -- Num Hidden Links matrices!!)
        WeightH2H=(REAL ***)Malloc(sizeof(REAL *)*(NumHL));
        DeltaH2H=(REAL ***)Malloc(sizeof(REAL *)*(NumHL));
        DeltaWeightH2H=(REAL ***)Malloc(sizeof(REAL *)*(NumHL));
        for(i=1; i<=NumHL; i++)   // for all hidden layers
        {
            snprintf(tempstring,sizeof(tempstring),"Number of units in layer %d",i-1);
            READINTFI(scratch,tempstring); // scratch contains the # of neurons in the hidden layer i-1
            nupl[i]=scratch; // Contains the number of units (sigmoid neurons) in this layer
            H2H[i-1]=(REAL **)makematr(NumPattern, nupl[i]+1,sizeof(REAL)); // matrix with NumPattern (#of training data) rows and
            DeltaH2H[i-1]=(REAL **)makematr(NumPattern, nupl[i]+1,sizeof(REAL)); // with #of units (sigmoid neurons) in layer i-1 columns
            DeltaWeightH2H[i-1]=(REAL **)makematr(nupl[i-1]+1,nupl[i]+1,sizeof(REAL));
            WeightH2H[i-1]=(REAL **)makematr(nupl[i-1]+1,nupl[i]+1,sizeof(REAL));

        }
    }

    /** OpenMP -- cumulative Delta for weights and biases (i,j) for output layer **/
    threadDeltaHO = (REAL ***)Malloc(sizeof(REAL *)*(nthreads)); // One matrix for thread
    /** OpenMP -- cumulative Delta for weights and biases (i,j) for hidden layers **/
    threadDeltaH2H = (REAL ***)Malloc(sizeof(REAL *)*(nthreads));
    for(i = 0 ; i < nthreads ; i++)
    {
        threadDeltaHO[i] = (REAL **)makematr(nupl[NumHL+1]+1,nupl[NumHL]+1,sizeof(REAL));
        threadDeltaH2H[i] = (REAL **)Malloc(sizeof(REAL *)*NumHL);
        for(h=1; h<=NumHL; h++)
        {
            threadDeltaH2H[i][h-1] = (REAL *)makevect((nupl[h-1]+1)*(nupl[h]+1),sizeof(REAL));
        }
    }
    /** OpenMP -- cumulative error **/
    threadError=makevect(nthreads,sizeof(REAL)); // OpenMP

    omp_set_num_threads(nthreads); // OpenMP

    /** H2H is for "hidden to hidden (layer)", HO is for "hidden to output" (layer) **/

    dimsum=nupl[1];
    for(i=1; i<=(NumHL+1); i++)   // All the layers but not the first (input layer)
    {
        dimsum=MAX(dimsum,nupl[i]);
    }
    Input=(REAL **)makematr(NumPattern, NumInput,sizeof(REAL)); // A matrix NumInput(784) X NumPattern (# of training data = 60.000)
    Target=(REAL **)makematr(NumPattern, NumOutput,sizeof(REAL)); // Like above...
    Sum=(REAL **)makematr(NumPattern, dimsum+1,sizeof(REAL));
    Output=(REAL **)makematr(NumPattern, NumOutput+1,sizeof(REAL));
    DeltaO=(REAL **)makematr(NumPattern, NumOutput+1,sizeof(REAL)); // Delta output
    WeightHO=(REAL **)makematr(nupl[NumHL]+1, NumOutput+1,sizeof(REAL));
    DeltaWeightHO=(REAL **)makematr(nupl[NumHL]+1, NumOutput+1,sizeof(REAL));

    SumDOW=makevect(dimsum+1,sizeof(REAL));
    ranpat=makevect(NumPattern,sizeof(int));

    ReadFromFile(InputFileName,Input,NumPattern,NumInput,sizeof(REAL)); // Populates the Input matrix by means of InputFileName
    ReadFromFile(TargetFileName,Target,NumPattern,NumOutput,sizeof(REAL)); // Populates the Target matrix by means of TargetFileName

    if(strcmp(LogFileName,NULLFILE))
    {
        fpl=Fopen(LogFileName,"w");
    }

    /** -----------------------------------------------INITIALIZATIONS--------------------------------------------------------------------------- **/

    for( j = 0 ; j <= nupl[NumHL] ; j++ )      /* initialize WeightHO and DeltaWeightHO */
    {
        for( k = 0 ; k <= nupl[NumHL+1] ; k ++ )
        {
            DeltaWeightHO[j][k] = 0.0 ;
            WeightHO[j][k] = 2.0 * ( drand48() - 0.5 ) * smallwt ; //smallwt -- Initialization scale (look at input.ml)
        }
    }

    for(h=NumHL; h>0; h--)
    {
        for(i=0; i<=nupl[h-1]; i++)
        {
            for(j=0; j<=nupl[h]; j++)
            {
                DeltaWeightH2H[h-1][i][j] = 0.0 ;
                WeightH2H[h-1][i][j] = 2.0 * ( drand48() - 0.5 ) * smallwt ;
            }
        }
    }

    if(strcmp(RestartFileName,NULLFILE))   // condition satisfied if strcmp != 0
    {
        fp=Fopen(RestartFileName,"r");
        if(strcmp(RestartDeltaFileName,NULLFILE))
        {
            fpd=Fopen(RestartDeltaFileName,"r");
        }
        for( k = 1 ; k <= nupl[NumHL+1] ; k ++ )
        {
            fscanf(fp,formatstring,&WeightHO[0][k]); // formatstring is %lf if REAL is double, else %f
            if(fpd) fscanf(fpd,formatstring,&DeltaWeightHO[0][k]);
            for( j = 1 ; j <= nupl[NumHL] ; j++ )
            {
                fscanf(fp,formatstring,&WeightHO[j][k]);
                if(fpd) fscanf(fpd,formatstring,&DeltaWeightHO[j][k]);
            }
            fscanf(fp,"\n");
            if(fpd) fscanf(fpd,"\n");
        }

        for(h=NumHL; h>0; h--)
        {
            for( j = 1 ; j <= nupl[h] ; j++ )
            {
                fscanf(fp,formatstring,&WeightH2H[h-1][0][j]);
                if(fpd) fscanf(fpd,formatstring,&DeltaWeightH2H[h-1][0][j]);
                for( i = 1 ; i <= nupl[h-1] ; i++ )
                {
                    fscanf(fp,formatstring,&WeightH2H[h-1][i][j]);
                    if(fpd) fscanf(fpd,formatstring,&DeltaWeightH2H[h-1][i][j]);
                }
                fscanf(fp,"\n");
                if(fpd)fscanf(fpd,"\n");
            }
        }

        if (fp) fclose(fp);
        if (fpd) fclose(fpd);
    }

    /** -----------------------------------------------------STOP INITIALIZATIONS------------------------------------------------------------ **/

    /** ------------------------------------------------------VERBOSE------------------------------------------------------------------------ **/

    if(verbose)
    {
        printf("\nInitial Bias and Weights\n");
        for( k = 1 ; k <=  nupl[NumHL+1] ; k ++ )
        {
            printf("Bias H to O[%d]: %f\n",k,WeightHO[0][k]); // BIASES OF NEURONS OF OUTPUT LAYER ARE IN THE ROW 0 OF THE MATRIX WeightHO,
            for( j = 1 ; j <= nupl[NumHL] ; j++ )             // so WeightHO[0][k]!!!
            {
                printf("Weight H[%d] to O[%d]: %f\n",j,k,WeightHO[j][k]);
            }
        }
        for(h=NumHL; h>0; h--)   // for all hidden layers
        {
            for( j = 1 ; j <= nupl[h] ; j++ )   // for all neurons in the actual layer
            {
                printf("Bias[%d][%d]: %f\n",h-1,j,WeightH2H[h-1][0][j]); // BIASES OF NEURONS IN THE ACTUAL LAYER
                for( i = 1 ; i <= nupl[h-1] ; i++ )   // for all neurons in the previous layer
                {
                    printf("Weight[%d][%d][%d]: %f\n",h-1,i,j,WeightH2H[h-1][i][j]);
                }
            }
        }
    }

    /** ---------------------------------------------------END VERBOSE----------------------------------------------------------------------- **/

    for( p = 0 ; p < NumPattern ; p++ )      /* initialize order of individuals */
    {
        ranpat[p] = p ;
    }

    /** --------------------------------------------------------START LEARNING--------------------------------------------------------------- **/

    for( epoch = 0 ; epoch < maxepoch ; epoch++)      /* iterate weight updates (through epochs) */
    {

        REAL tempSum = 0.0;
        Error = 0.0 ; // before starting to consider the training data

        startime = omp_get_wtime();

        #pragma omp parallel private(np,p,i,j,h,k,j_delta,i_delta,k_delta)
        {
            int id = omp_get_thread_num(); // me
            int nthreads = omp_get_num_threads(); // how many threads
            REAL temp; // Contains temporary sums

            /** ----------------------------------RANDOMIZE TRAINING SET (RANPAT)------------------------**/
#if defined(RANDOMIZE_INDIVIDUALS)
            for( p = id ; p <  NumPattern ; p+=nthreads )      /* randomize order of individuals */
            {
                ranpat[p] = p ;
            }

            #pragma omp barrier

            #pragma omp single
            {
                for( p = 0 ; p < NumPattern-1 ; p++)
                {
                    np = (p+1) + (rand()%(NumPattern - p -1)) ;
                    op = ranpat[p] ;
                    ranpat[p] = ranpat[np] ;
                    ranpat[np] = op ; // random swap
                }
            }//Implicit barrier here

#endif
            /** ---------------------------------END RANDOMIZATION--------------------------------------**/

            /** START ITERATING THE TRAINING EXAMPLES **/
            for( np = id ; np < NumPattern ; np+=nthreads )
            {

                p = ranpat[np]; // A random input in training set -- FIXED FOR EACH ITERATION OF PREVIOUS FOR CYCLE

                /** FEEDFORWARD **/

                for( j = 1 ; j <= nupl[1]; j++ )      /* compute hidden unit activations */
                {
                    temp = WeightH2H[0][0][j] ; // WeightH2H[k][i][j] with 0 <= k <= NumHL -1 . i=0 is for biases of neurons in j
                    for( i = 1 ; i <= nupl[0] ; i++ )
                    {
                        temp += Input[p][i-1] * WeightH2H[0][i][j] ; /* Matrix SumH = Input_Matrix x Weight_Input_Matrix
                                                                      The Input_Matrix has one row per sample
                                                                      The Weight_Input_Matrix has one row per input
                                                                      The SumH Matrix is initialized with the Bias */
                    }
                    Sum[p][j] = temp;
                    H2H[0][p][j] = 1.0/(1.0 + exp(-temp)) ;    /* Compute the sigmoid of all the elements of SumH (activation of neurons in layer 1) */
                }


                for( h=1; h<NumHL; h++)
                {
                    for(k=1; k<=nupl[h+1]; k++)
                    {
                        temp = WeightH2H[h][0][k];
                        for( j = 1 ; j <= nupl[h] ; j++ )
                        {
                            temp += H2H[h-1][p][j] * WeightH2H[h][j][k] ;/* Matrix SumO = Hidden_Matrix x Weight_Output Matrix
                                                                      The Hidden_Matrix has one row per sample
                                                                      The Weight_Output_Matrix has one row per number of neurons in the hidden level
                                                                      The SumO Matrix is initialized with the Bias */
                        }
                        Sum[p][k] = temp;
                        H2H[h][p][k] = 1.0/(1.0 + exp(-temp)) ;   /* Sigmoidal Outputs *//* Compute the sigmoid of all the elements of SumO */
                    }
                }


                for( k = 1 ; k <= nupl[NumHL+1] ; k++ )      /* compute output unit activations and errors */
                {
                    temp = WeightHO[0][k] ;
                    for( j = 1 ; j <= nupl[NumHL] ; j++ )
                    {
                        temp += H2H[NumHL-1][p][j] * WeightHO[j][k] ;/* Matrix SumO = Hidden_Matrix x Weight_Output Matrix
                                                                      The Hidden_Matrix has one row per sample
                                                                      The Weight_Output_Matrix has one row per number of neurons in the hidden level
                                                                      The SumO Matrix is initialized with the Bias */
                    }
                    Sum[p][k] = temp;
                    /** END FEEDFORWARD **/

                    Output[p][k] = 1.0/(1.0 + exp(-temp)) ;   /* Sigmoidal Outputs *//* Compute the sigmoid of all the elements of SumO */
                    /*              printf("Epoch %d, pattern %d, output %d, output %f, target %f\n", epoch, p, k, Output[p][k], Target[p][k-1]); */
                    /*              Output[p][k] = SumO[p][k];      Linear Outputs */

                    /** COMPUTE OUTPUT ERROR **/
                    threadError[id] += 0.5 * (Target[p][k-1] - Output[p][k]) * (Target[p][k-1] - Output[p][k]) ;   /* SSE */
                    /*              Error -= ( Target[p][k-1] * log( Output[p][k] ) + ( 1.0 - Target[p][k-1] ) * log( 1.0 - Output[p][k] ) ) ;    Cross-Entropy Error */

                    /** THIS IS THE OUTPUT ERROR IN OUTPUT NEURON K, FOR THE TRAINING EXAMPLE P **/
                    DeltaO[p][k] = (Target[p][k-1] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   /* Sigmoidal Outputs, SSE */
                    /* derivative of the error x derivative of the sigmoidal function */
                    /*              DeltaO[p][k] = Target[p][k-1] - Output[p][k];     Sigmoidal Outputs, Cross-Entropy Error */
                    /*              DeltaO[p][k] = Target[p][k-1] - Output[p][k];     Linear Outputs, SSE */

                    /** END OUTPUT ERROR **/

                    /** HERE DELTA WEIGHTS AND BIASES FOR OUTPUT LAYER**/
                    for(j_delta = 0; j_delta <= nupl[NumHL] ; j_delta++)
                    {

                        if(id == 0 && np == 0)  // Only one thread does that (id == 0) and at first (when np == id == 0)
                        {
                            threadDeltaHO[id][k][j_delta] = alpha * DeltaWeightHO[j_delta][k]; //inizialization
                        }

                        threadDeltaHO[id][k][j_delta] += eta * ((j_delta>0)?H2H[NumHL-1][p][j_delta]:1) * DeltaO[p][k];
                        /* Matrix DeltaWeightHO = eta x (Trasposte of Hidden_Matrix) x DeltaO */
                        /* DeltaWeight is initialized with the "momentum" */
                    }
                }

                /** START BACK PROPAGATION OF THE ERROR (DELTA) **/
                for( j = 1 ; j <= nupl[NumHL] ; j++ )      /* 'back-propagate' errors to hidden layer */
                {
                    temp = 0.0 ;
                    for( k = 1 ; k <= nupl[NumHL+1] ; k++ )
                    {
                        temp += WeightHO[j][k] * DeltaO[p][k];
                    }
                    DeltaH2H[NumHL-1][p][j] = temp * H2H[NumHL-1][p][j] * (1.0 - H2H[NumHL-1][p][j]) ;

                    for(i_delta = 0; i_delta <= nupl[NumHL-1]; i_delta++)
                    {

                        if(id == 0 && np == 0)
                        {
                            threadDeltaH2H[id][NumHL-1][j*(nupl[NumHL-1]+1)+i_delta] = alpha *  DeltaWeightH2H[NumHL-1][i_delta][j];
                        }

                        threadDeltaH2H[id][NumHL-1][j*(nupl[NumHL-1]+1)+i_delta] += eta * ((i_delta==0)?1:H2H[NumHL-2][p][i_delta]) * DeltaH2H[NumHL-1][p][j];

                    }

                }


                for(h=NumHL-1; h>0; h--)
                {
                    for( j = 1 ; j <= nupl[h] ; j++ )
                    {
                        temp = 0.0 ;
                        for( k = 1 ; k <= nupl[h+1] ; k++ )
                        {
                            //Qui prima c'era WeightH2H[h-1][j][k]..che fosse un errore?..
                            temp += WeightH2H[h][j][k] * DeltaH2H[h][p][k]; // temp is a part of DeltaH2H[h-1][p][j]
                        }
                        DeltaH2H[h-1][p][j] = temp * H2H[h-1][p][j] * (1.0 - H2H[h-1][p][j]) ;


                        for( i_delta = 0 ; i_delta <= nupl[h-1] ; i_delta++)
                        {

                            /** ........................................ **/
                            if(id == 0 && np == 0)  // Only one thread does that (id == 0) and at first (when np == id == 0)
                            {
                                threadDeltaH2H[id][h-1][j*(nupl[h-1]+1)+i_delta] = alpha * DeltaWeightH2H[h-1][i_delta][j];//inizialization
                            }

                            if(h>1)
                                threadDeltaH2H[id][h-1][j*(nupl[h-1]+1)+i_delta] += eta * ((i_delta==0)?1:H2H[h-2][p][i_delta]) * DeltaH2H[h-1][p][j];
                            else
                                threadDeltaH2H[id][0][j*(nupl[0]+1)+i_delta] += eta * ((i_delta==0)?1:Input[p][i_delta-1]) * DeltaH2H[0][p][j];

                            /* Matrix DeltaWeightIH = eta x (Trasposte of Input_Matrix) x DeltaH */
                            /* DeltaWeight is initialized with the "momentum" */
                        }


                    }
                    /** HERE DELTA WEIGHTS AND BIASES FOR HIDDEN LAYERS (h>=1)**/

                }
                /** END BACK PROPAGATION OF THE ERROR (DELTA) **/

            }
            /** END ITERATING THE TRAINING EXAMPLES (for one thread)**/
        } // End of parallel region (implicit barrier here)

        /** Done by a single thread (REDUCTION PHASE -- Error)**/
        for(i = 0 ; i < nthreads ; i++)
        {
            Error += threadError[i];
            threadError[i] = 0.0;
        }

        /** Done by a single thread (REDUCTION PHASE -- Delta)**/
        /** Output layer (h = NumHL+1)**/
        for(i = 0 ; i <= nupl[NumHL] ; i++)
        {
            for(j = 1 ; j <= nupl[NumHL+1] ; j++)
            {
                tempSum = 0.0;
                for(k = 0 ; k < nthreads ; k++)
                {
                    tempSum += threadDeltaHO[k][j][i];
                    threadDeltaHO[k][j][i] = 0.0;
                }
                DeltaWeightHO[i][j] = tempSum;
                /** GRADIENT DESCENT **/
                WeightHO[i][j] += tempSum;/* update weights WeightHO */
            }
        }

        /** Done by a single thread (REDUCTION PHASE -- Delta)**/
        /** h>0 **/
        for(h=NumHL; h>0; h--)
        {
            for( j = 1 ; j <= nupl[h] ; j++ )
            {
                for( i = 0 ; i <= nupl[h-1] ; i++ )
                {

                    tempSum = 0.0;
                    for( k = 0 ; k < nthreads ; k++)
                    {
                        tempSum += threadDeltaH2H[k][h-1][j*(nupl[h-1]+1)+i];
                        threadDeltaH2H[k][h-1][j*(nupl[h-1]+1)+i] = 0.0;
                    }
                    DeltaWeightH2H[h-1][i][j] = tempSum;
                    /** GRADIENT DESCENT **/
                    WeightH2H[h-1][i][j] += tempSum;
                }
            }
        }

        fprintf(stdout, "Elapsed time for epoch %-5d : %lf\n", epoch, omp_get_wtime() - startime);

        fprintf(stdout, "Epoch %-5d :   Error = %f\n", epoch, Error) ;
        if(fpl)
        {
            fprintf(fpl,"Epoch %-5d :   Error = %f\n", epoch, Error);
            fflush(fpl);
        }
        /** Eps is the error threshold **/
        if( Error < Eps ) break ;  /* stop learning when 'near enough' */
    }

    /** -------------------------------------------STOP LEARNING (END EPOCHS OR NEAR ENOUGH)------------------------------------------------- **/



#if 0
    fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d\n\nPat\t", epoch) ;   /* print network outputs */
    for( i = 0 ; i < NumInput ; i++ )
    {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 0 ; k < NumOutput ; k++ )
    {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 0 ; p < NumPattern ; p++ )
    {
        fprintf(stdout, "\n%d\t", p) ;
        for( i = 0 ; i < NumInput ; i++ )
        {
            fprintf(stdout, "%f\t", Input[p][i]) ;
        }
        for( k = 1 ; k <= NumOutput ; k++ )
        {
            fprintf(stdout, "%f\t%f\t", Target[p][k-1], Output[p][k]) ;
        }
    }
#endif
    if(verbose)
    {
        printf("\nFinal Bias and Weights\n");
    }
    fp=Fopen(ResultFileName,"w");
    fpd=Fopen(DeltaFileName,"w");
    for( k = 1 ; k <= NumOutput ; k ++ )
    {
        if(verbose)
        {
            printf("Bias H to O[%d]: %f\n",k,WeightHO[0][k]);
        }
        fprintf(fp,"%7.5f ",WeightHO[0][k]);
        fprintf(fpd,"%g ",DeltaWeightHO[0][k]);
        for( j = 1 ; j <= nupl[NumHL] ; j++ )
        {
            if(verbose)
            {
                printf("Weight H[%d] to O[%d]: %f\n",j,k,WeightHO[j][k]);
            }
            fprintf(fp,"%7.5f ",WeightHO[j][k]);
            fprintf(fpd,"%g ",DeltaWeightHO[j][k]);
        }
        fprintf(fp,"\n");
        fprintf(fpd,"\n");
    }
    for(h=NumHL; h>0; h--)
    {
        for( j = 1 ; j <= nupl[h] ; j++ )
        {
            if(verbose)
            {
                printf("BiasH2H[%d][%d]: %f\n",h,j,WeightH2H[h-1][0][j]);
            }
            fprintf(fp,"%7.5f ",WeightH2H[h-1][0][j]);
            fprintf(fpd,"%g ",DeltaWeightH2H[h-1][0][j]);
            for( i = 1 ; i <= nupl[h-1] ; i++ )
            {
                if(verbose)
                {
                    printf("WeightH2H[%d][%d] to H{%d]: %f\n",h,i,j,WeightH2H[h-1][i][j]);
                }
                fprintf(fp,"%7.5f ",WeightH2H[h-1][i][j]);
                fprintf(fpd,"%g ",DeltaWeightH2H[h-1][i][j]);
            }
            fprintf(fp,"\n");
            fprintf(fpd,"\n");
        }
    }
    if(fp) fclose(fp);
    if(fp) fclose(fpd);
    if(fpl) fclose(fpl);
    return 0 ;
}

/*******************************************************************************/
