/*
      Parallel and Distributed Systems
      \file   v2.c
      \brief  Implementation for the Ising Model in CUDA
              One thread computing a block of moments

      \authors Ioannis Gonidelis       Dimitra Karatza
      \AEMs     8794                    8828
      \date   2020-01-15
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define BLOCK_DIMENSION  11
#define GRID_DIMENSION 47
#define N 517 //dimention


void validation(int n,int k,int *expected,int *G){
  int flag=0;
  for(int v = 0; v < n*n; v++){
    if(expected[v] != G[v]){
      flag=-1;
      break;
    }
  }
  if(flag==0){
    printf("\033[0;32m");
    printf("k=%d: CORRECT ISING MODEL",k);
    printf("\033[0m \n");
  }else{
    printf("k=%d: WRONG ISING MODEL\n",k);
  }
}



__global__ void calc_moment(int *G, int* newG, double* w, int n){

  //NOTE: gridDim.x = gridDim.y it's the same

  int fit = n/(gridDim.x*BLOCK_DIMENSION);   //number of complete blocks that fit into G

  //Global G indices
  int ix=threadIdx.x+blockIdx.x*blockDim.x;
  int iy=threadIdx.y+blockIdx.y*blockDim.y;

  int x,y;//shared memory indices

  int thread_step_x= blockDim.x*gridDim.x;

  double infl; //influence of neighbors on current moment

  for(int i=0; i<(fit+1)*(fit+1); i++){
    infl=0;
    if(ix<N && iy<N){
      //for all the neighbors
      for(int c=0;c<5;c++){
        for(int d=0;d<5;d++){
          //Do not update if the next neighbor coincides with the current point
          if((c!=2) || (d!=2)){

            //Windows centered on the edge lattice points wrap around to the other side
            y = ((c-2)+iy+n) % n;
            x = ((d-2)+ix+n) % n;

            //Influence of a neighbor is increased
            //Add to infl the weight*value of the previous neighbor
            infl += G[y*n+x] * w[c*5+d];

          }
        }
      }
      //Next value of a moment is defined according to the value of infl
      if(infl>0.0001){
        newG[iy*n+ix]=1;
      }else if(infl<-0.0001){
        newG[iy*n+ix]=-1;
      }else{
        newG[iy*n+ix]=G[iy*n+ix];
      }
    }

    //update G coordinates - traverse horizontally though G map
    if((ix+thread_step_x)/n>=1){
      iy=blockDim.y*gridDim.y+iy;
    }else{
      iy=iy;
    }
    ix=(ix+thread_step_x)%n;
  }
}


void ising( int *G, double *w, int k, int n){

  int *newG,*swapG;
  cudaMallocManaged(&newG,n*n*sizeof(int)); //save previous G before changing it

  dim3 block(BLOCK_DIMENSION, BLOCK_DIMENSION);
  int grid_dimension = GRID_DIMENSION; //define it gloabaly or find a way to produce it
  dim3 grid(grid_dimension, grid_dimension);

  //for every iteration (k)
  for(int t=0;t<k;t++){

    //Call kernel function
    calc_moment<<<grid,block>>>(G, newG, w,n);

    // Synchronize threads before swapping the arrays
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    //Swap arrays G and newG
    swapG=newG;
    newG=G;
    G=swapG;
  }

  //If last k is an odd number, then the returned G should be newG
  if(k % 2 == 1){
    memcpy(newG, G, n*n*sizeof(int));
  }

}


int main(void)
{
  //k = number of iterations
  int k = 1;
  int n=N;

  // Array of weights
  double *weights;
  cudaMallocManaged(&weights,5*5*sizeof(double));
  double w[25] = {0.004, 0.016, 0.026, 0.016, 0.004,
                  0.016, 0.071, 0.117, 0.071, 0.016,
                  0.026, 0.117, 0, 0.117, 0.026,
                  0.016, 0.071, 0.117, 0.071, 0.016,
                  0.004, 0.016, 0.026, 0.016, 0.004};
  memcpy(weights,w,sizeof(w));


  // Get the moments of array G from the binary file
  FILE *fptr = fopen("conf-init.bin","rb");
  if (fptr == NULL){
      printf("Error: Cannnot open file");
      exit(1);
  }
  int *G;
  cudaMallocManaged(&G,n*n*sizeof(int));
  fread(G, sizeof(int), n*n, fptr);
  fclose(fptr);


  //Save a copy of G to call again function ising() for different k
  //because ising() is changing the array G
  int *copyG;
  cudaMallocManaged(&copyG,n*n*sizeof(int));
  memcpy(copyG, G, n*n*sizeof(int));


  //Call ising for k=1
  ising(G, weights, k, n);
   // Check results by comparing with ready data for k=1
  int *expected;
  cudaMallocManaged(&expected,n*n*sizeof(int));
  fptr = fopen("conf-1.bin","rb");
  if (fptr == NULL){
      printf("Error: Cannnot open file");
      exit(1);
  }
  fread(expected, sizeof(int), n*n, fptr);
  fclose(fptr);
  validation(n,k,expected,G);

  //Call ising for k=4
  k=4;
  memcpy(G, copyG, n*n*sizeof(int));
  ising(G, weights, k, n);
  // Check for k = 4
  fptr = fopen("conf-4.bin","rb");
  if (fptr == NULL){
      printf("Error: Cannnot open file");
      exit(1);
  }
  fread(expected, sizeof(int), n*n, fptr);
  fclose(fptr);
  validation(n,k,expected,G);


  //Call ising for k=11;
  k=11;
  memcpy(G, copyG, n*n*sizeof(int));
  ising(G, weights, k, n);
  // Check for k = 11
  fptr = fopen("conf-11.bin","rb");
  if (fptr == NULL){
      printf("Error: Cannnot open file");
      exit(1);
  }
  fread(expected, sizeof(int), n*n, fptr);
  fclose(fptr);
  validation(n,k,expected,G);

  return 0;

}
