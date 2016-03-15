#include "counting.h"
#include <cstdio>
#include <cassert>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
using namespace std;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void build(const char *text,int *pos, int *buffer, int text_size, int *offset){
	int idx =threadIdx.x + blockDim.x*blockIdx.x;
	if (text[idx]>=int('a') && text[idx]<=int('z'))
		buffer[idx]=1;
	else
		buffer[idx]=0;
	int level=0;
	int offset_now=0;
	int offset_next=0;
	while (((idx%2) ==0)){
		offset_now=offset[level];
		level++;
		offset_next=offset[level];
		buffer[offset_next+idx/2]=(buffer[offset_now+idx]>0 && buffer[offset_now+idx+1]>0)?(buffer[offset_now+idx]+buffer[offset_now+idx+1]):0;
		__syncthreads();
		idx=idx/2;
		if(idx==0)
			break;
	}
	idx = threadIdx.x + blockDim.x*blockIdx.x;
	int head=idx;
	level=0;
	while(1){
		offset_now=offset[level];
		if(buffer[offset_now+head]!=0){
			if(head==0){
				break;
			}
				
			else{
				head=(head-1)/2;
				level++;
			}
		}
		else
			break;
	}

	while(level!=0){
		level=level-1;
		offset_now=offset[level];
		if(buffer[offset_now+head*2+1]==0){
			head=head*2+1;
		}
		else{
			head=head*2;
		}
	}

	pos[idx]=(head==0)?idx-head+1:idx-head;
}



void CountPosition(const char *text, int *pos, int text_size)
{
	int Threadperblock = 256;
	int Blockpergird = (text_size - Threadperblock + 1) / Threadperblock;
	int *buffer;
	int count=0;
	int size=text_size;
	while(size!=0){
		size=size/2;
		count++;
	}
	int *offset;
	int *offset_cu;
	offset=(int *)malloc(50*sizeof(int));
	int i=1;
	size=text_size;
	offset[0]=0;
	while(size!=0){
		offset[i]=offset[i-1]+size;
		size=size/2;
		i++;
	}
	cudaMalloc(&offset_cu,sizeof(int)*count);
	cudaMemcpy(offset_cu,offset,sizeof(int)*count,cudaMemcpyHostToDevice);
	cudaMalloc(&buffer,sizeof(int)*text_size*2);
	build << <Blockpergird, Threadperblock >> >(text,pos, buffer ,text_size,offset_cu);
	cudaDeviceSynchronize();
	cudaFree(buffer);
	cudaFree(offset_cu);
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO

	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
