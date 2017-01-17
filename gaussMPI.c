
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>


/*************************************************
 *  *  Globle Variables
 *   *************************************************/

int N=2000;  	/* Matrix size */
int procs;  /* Number of processors to use */
int myid;	/* Process Rank */

/* Matrices and vectors */
float  **A,*x;
/* A * X = B, solve for X */

/*******************************************************
 *  * Gauss Elimination MPI Implementation Function
 *   *******************************************************/
void gaussEliminationMPI();

/******************************************************
 *  * Generate random values for provided order mattrix
 *   ******************************************************/
void generateMatrix(){
	int i,j;
	for (i=0; i<N; i++){
		for (j=0; j<N+1; j++){
			A[i][j] = (rand()/50000);
		}
	}

	/*A[4][5]={
 * 			{2,1,-1,2,5},
 * 						{4,5,-3,6,9},
 * 									{-2,5,-2,6,4},
 * 												{4,11,-4,8,2}};*ans 3 1 -2 1*/

			/*A[0][0]=2;
 * 			A[0][1]=1;
 * 						A[0][2]=-1;
 * 									A[0][3]=2;
 * 												A[0][4]=5;
 * 															A[1][0]=4;
 * 																		A[1][1]=5;
 * 																					A[1][2]=-3;
 * 																								A[1][3]=6;
 * 																											A[1][4]=9;
 * 																														A[2][0]=-2;
 * 																																	A[2][1]=5;
 * 																																				A[2][2]=-2;
 * 																																							A[2][3]=6;
 * 																																										A[2][4]=4;
 * 																																													A[3][0]=4;
 * 																																																A[3][1]=11;
 * 																																																			A[3][2]=-4;
 * 																																																						A[3][3]=8;
 * 																																																									A[3][4]=2;*/

		/*A[0][0]=3;
 * 		A[0][1]=2;
 * 				A[0][2]=-4;
 * 						A[0][3]=3;
 * 								A[1][0]=2;
 * 										A[1][1]=3;
 * 												A[1][2]=3;
 * 														A[1][3]=15;
 * 																A[2][0]=5;
 * 																		A[2][1]=-3;
 * 																				A[2][2]=1;
 * 																						A[2][3]=14;*/
}

/***********************************************************
 *  * print Matrix if required.large matrix will not be printed.
 *  ***********************************************************/
void printMarix(char* msg){
	int i,j;
	printf("%s\n",msg);
		for(i=0;i<N;i++)
		{
			printf("%d\t",i);
			for(j=0;j<N+1;j++)
			{
				printf("%lf\t",A[i][j]);
			}
			printf("\n");
		}
}

/************************************************
 *  * Print the final output of Linear Equation.
 *   ************************************************/
void printOutput(){
	int i;
	for(i=0;i<N;i++){
		printf("X[%d]-->%f\n",i,x[i]);
	}
}

/************************************************************
 *  * Back Substitution implementation after matrix transform to
 *   * upper triangular form
 *    ************************************************************/
void backSubstitution(){
	double sum;
	int i,j;
	x[N-1]=A[N-1][N]/A[N-1][N-1];

		for(i=N-2; i>=0; i--)
		{
			sum=0;
			for(j=i+1; j<=N-1; j++)
			{
				sum=sum+A[i][j]*x[j];
			}
			x[i]=(A[i][N]-sum)/A[i][i];
		}
}

/***********************************************************
 *  * Main finction for Gauss Elimination in MPI
 *   ***********************************************************/

int main(int argc, char **argv) {

	int q;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    printf("\nProcess number %d", myid);

    /***********Dynamic Allowcation of matrix and X vector***********/
	A = (float **)calloc(N,sizeof(float*));
	for (q=0; q < N; q++)
		A[q] = (float*)calloc(N+1,sizeof(float*));

	/************Allocate Vector 'x for Pthread Answer8***************/
	x = (float*) malloc(sizeof(float)*N);
	/****************************************************************/

    /********** Generate Random Matrix A with vector B Combineds**************/
    if (myid == 0) {
        generateMatrix();

        /* Print Initial matrices */
       /* printMarix("Initial Matrix:-");*/
    }
    /******************************************************/

    /********* Gaussian Elimination **********/
    gaussEliminationMPI();
    /***************************************/

    /********* Back substitution *********************/
    if (myid == 0) {

    	backSubstitution();

    	/**Print Final Output**/
    	/*	printOutput();*/
    }
    /**************************************************/
    free(A);
    free(x);

    MPI_Finalize();
    return 0;
}

/*******************************************************
 *  *Gauss Elimination without pivoting
 *   ******************************************************/

void gaussEliminationMPI() {

	MPI_Status status;
	MPI_Request request;

    /*printf("\nProcessor %d in Gauss Function...:- My Rank:-%d\n",myid,myid);*/
    int norm, row, col;
    float multiplier;
    int i,j;

    /*Timing Parameter*/
    double starttime = 0.0, endtime;


    if (myid == 0) {
        printf("\nParallel Computing Started Using MPI...Process=%d\n",myid);
        starttime = MPI_Wtime();
    }


    /*MPI_Barrier(MPI_COMM_WORLD);*/

    /* Gaussian elimination */
    for (norm = 0; norm < N-1; norm++) {
        /* Broadcast norm row in every step to all processes.*/
    	MPI_Bcast(&(A[norm][0]), N+1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    	if(myid==0){
    		/*Gaussian elimination*/
    		for (i = 1; i < procs; i++) {
			/*using static interleaved scheduling data sent to corresponding process*/
				for (row = norm + 1 + i; row < N; row += procs) {
					MPI_Send(&(A[row][0]), N+1, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
					/*printf("\nrow %d To processor %d \n",row,i);*/
				}
			}
    	}

    	/*MPI_Barrier(MPI_COMM_WORLD);*/

			if (myid == 0) {
				/*printf("Process ID = %d",myid);*/
				/*printMarix("Process- 0 Initial Matrix:-");*/

				for (row = norm + 1; row < N; row += procs) {
					multiplier = A[row][norm] / A[norm][norm];
					for (col = norm; col < N+1; col++) {
						A[row][col] -= A[norm][col] * multiplier;
					}
				}

				/*printMarix("After one iteration  0 Initial Matrix:-");*/
				/*Receive the updated data from other processes*/
				for (i = 1; i < procs; i++) {
					for (row = norm + 1 + i; row < N; row += procs) {
						MPI_Recv(&A[row][0], N+1, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
					}
				}
				/*printMarix("My rank 0 Final Matrix:-");*/
				if (norm == N - 2) {
					endtime = MPI_Wtime();
					printf("elapsed time = %f\n", endtime - starttime);
				}
			}
			else {
				/*printf("/nProcess :- %d in else My Rank/n",myid,myid);*/
				for (row = norm + 1 + myid; row < N; row += procs) {
					/*printf("\nProcess :- %d in else For , my id=%d\n",myid,myid);*/

					MPI_Recv(&(A[row-1][0]),N+1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
					multiplier = A[row-1][norm] / A[norm][norm];
					for (col = norm; col < N+1; col++) {
						A[row-1][col] -= A[norm][col] * multiplier;
					}
					MPI_Send(&(A[row-1][0]), N+1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);

				}
			}
			/*Barrier syncs all processes*/
			        /*MPI_Barrier(MPI_COMM_WORLD);*/

    	}

    if (myid == 0) {
        endtime = MPI_Wtime();
        printf("\nelapsed time = %f\n", endtime - starttime);

    }

}

