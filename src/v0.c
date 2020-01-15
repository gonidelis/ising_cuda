/*
      Parallel and Distributed Systems
      \file   v0.c
      \brief  Serial Implementation for the Ising Model

      \authors Ioannis Gonidelis       Dimitra Karatza
      \AEMs     8794                    8828
      \date   2020-01-15
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int n=517; // n = dimentions

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

void ising( int *G, double *w, int k, int n){

  int *newG=calloc(n*n,sizeof(int)); //save previous G before changing it
  int *swapG; //temporary variable useful for swapping the arrays G and newG
  double infl=0; //temporary value to define the influence of the neighbors and the new value of each moment
  int x,y; //indices of a moment

  //for every iteration (k)
  for(int t=0;t<k;t++){

    //for every moment of G (n*n)
    for(int i=0;i<n;i++){ //for every y
      for(int j=0;j<n;j++){ //for every x

        infl=0;

        //for all the neighbors
        for(int c=0;c<5;c++){
					for(int d=0;d<5;d++){

            //Do not update if the next neighbor coincides with the current point
            if((c!=2) || (d!=2)){

              //Windows centered on the edge lattice points wrap around to the other side
              y = ((c-2)+i+n) % n;
  					  x = ((d-2)+j+n) % n;

              //Influence of a neighbor is increased
              //Add to infl the weight*value of the previous neighbor
  					  infl += G[y*n+x] * w[c*5+d];

            }
          }
        }

        //Next value of a moment is defined according to the value of infl
        if(infl>0.0001){
          newG[i*n+j]=1;
        }else if(infl<-0.0001){
          newG[i*n+j]=-1;
        }else{ //delete this
          newG[i*n+j]=G[i*n+j];
        }
      }
    }

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


int main(){

	int k = 1; // k = number of iterations

  // Array of weights
  double weights[] = {0.004, 0.016, 0.026, 0.016, 0.004,
                      0.016, 0.071, 0.117, 0.071, 0.016,
                      0.026, 0.117, 0, 0.117, 0.026,
                      0.016, 0.071, 0.117, 0.071, 0.016,
                      0.004, 0.016, 0.026, 0.016, 0.004};


	// Get the moments of array G from the binary file
  FILE *fptr = fopen("conf-init.bin","rb");
  if (fptr == NULL){
      printf("Error: Cannnot open file");
      exit(1);
  }
  int *G = calloc(n*n, sizeof(int));
  fread(G, sizeof(int), n*n, fptr);
  fclose(fptr);


  //Save a copy of G to call again function ising() for different k
  //because ising() is changing the array G
  int *copyG = calloc(n*n, sizeof(int));
  memcpy(copyG, G, n*n*sizeof(int));


  //Call ising for k=1
  ising(G, weights, k, n);
	// Check results by comparing with ready data for k=1
	int *expected = calloc(n*n, sizeof(int));
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
