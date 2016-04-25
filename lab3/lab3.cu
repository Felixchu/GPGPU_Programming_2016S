#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void CalculateFixed(const float *background,const float* target,const float * mask,float * fixed,const int wb,const int hb,const int wt,const int ht,const int oy,const int ox){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	int Nt=curt-wt; int Nt_m=Nt; if(yt==0) Nt_m=0; if(yt==0) Nt=curt;
	int St=curt+wt; int St_m=St; if(yt==ht-1) St_m=0; if(yt==ht-1) St=curt;
	int Wt=curt-1;  int Wt_m=Wt; if(xt==0) Wt_m=0; if(xt==0) Wt=curt;
	int Et=curt+1;  int Et_m=Et; if(xt==wt-1) Et_m=0; if(xt==wt-1) Et=curt;

	if (yt < ht and xt < wt and mask[curt] > 127.0f ) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		int Nb=curb-wb;  if(Nb<0) Nb=curb;
		int Sb=curb+wb;  if(Sb>=hb) Sb=curb;
		int Eb=curb+1;   if(Eb>=wb) Eb=curb;
		int Wb=curb-1;   if(Wb<0)  Wb=curb;
		
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			
			if(mask[Nt_m]> 127.0f and mask[St_m]> 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]> 127.0f){
				fixed[curt*3+0] = 4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0]);
				fixed[curt*3+1] = 4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1]);
				fixed[curt*3+2] = 4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2]);
		
			}
			else if(mask[Nt_m]< 127.0f and mask[St_m]> 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2];
		
			}
			else if(mask[Nt_m]> 127.0f and mask[St_m]< 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Sb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Sb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Sb*3+2];
			
			}
			else if(mask[Nt_m]> 127.0f and mask[St_m]> 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Wb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Wb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Wb*3+2];
	
			}
			else if(mask[Nt_m]> 127.0f and mask[St_m]> 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Eb*3+2];
	
			}
			else if(mask[Nt_m]< 127.0f and mask[St_m]< 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Sb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Sb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Sb*3+2];
		
			}
			else if(mask[Nt_m]< 127.0f and mask[St_m]> 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Wb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Wb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Wb*3+2];
		
			}
			else if(mask[Nt_m]< 127.0f and mask[St_m]> 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Eb*3+2];
			
			}
			else if(mask[Nt_m]> 127.0f and mask[St_m]< 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Sb*3+0]+background[Wb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Sb*3+1]+background[Wb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Sb*3+2]+background[Wb*3+2];
	
			}
			else if(mask[Nt_m]> 127.0f and mask[St_m]< 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Sb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Sb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Sb*3+2]+background[Eb*3+2];

			}
			else if(mask[Nt_m]> 127.0f and mask[St_m]> 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Wb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Wb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Wb*3+2]+background[Eb*3+2];

			}
			else if(mask[Nt_m]< 127.0f and mask[St_m]< 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Sb*3+0]+background[Wb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Sb*3+1]+background[Wb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Sb*3+2]+background[Wb*3+2];
	
			}
			else if(mask[Nt_m]< 127.0f and mask[St_m]< 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Sb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Sb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Sb*3+2]+background[Eb*3+2];

			}
			else if(mask[Nt_m]< 127.0f and mask[St_m]> 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Wb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Wb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Wb*3+2]+background[Eb*3+2];

			}
			else if(mask[Nt_m]> 127.0f and mask[St_m]< 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Sb*3+0]+background[Wb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Sb*3+1]+background[Wb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Sb*3+2]+background[Wb*3+2]+background[Eb*3+2];

			}
			else{
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Wb*3+0]+background[Sb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Wb*3+1]+background[Sb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Wb*3+2]+background[Sb*3+2]+background[Eb*3+2];

			}
		}
	}
}


__global__ void PoissonImageCloningIteration(const float *fixed,const float* mask,const float * buf1,float * buf2, const int wt, const int ht){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
		int Nt=curt-wt; int Nt_m=Nt; if(yt==0) Nt_m=0; if(yt==0) Nt=curt;
	int St=curt+wt; int St_m=St; if(yt==ht-1) St_m=0; if(yt==ht-1) St=curt;
	int Wt=curt-1;  int Wt_m=Wt; if(xt==0) Wt_m=0; if(xt==0) Wt=curt;
	int Et=curt+1;  int Et_m=Et; if(xt==wt-1) Et_m=0; if(xt==wt-1) Et=curt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f ) {
		if(mask[Nt_m]> 127.0f and mask[St_m]> 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[St*3+0]+buf1[Wt*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[St*3+1]+buf1[Wt*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[St*3+2]+buf1[Wt*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt_m]< 127.0f and mask[St_m]> 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[St*3+0]+buf1[Wt*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[St*3+1]+buf1[Wt*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[St*3+2]+buf1[Wt*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt_m]> 127.0f and mask[St_m]< 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[Wt*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[Wt*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[Wt*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt_m]> 127.0f and mask[St_m]> 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[St*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[St*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[St*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt_m]> 127.0f and mask[St_m]> 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[St*3+0]+buf1[Wt*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[St*3+1]+buf1[Wt*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[St*3+2]+buf1[Wt*3+2]))/4;
		}
		else if(mask[Nt_m]< 127.0f and mask[St_m]< 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Wt*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Wt*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Wt*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt_m]< 127.0f and mask[St_m]> 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(+buf1[St*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(+buf1[St*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(+buf1[St*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt_m]< 127.0f and mask[St_m]> 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[St*3+0]+buf1[Wt*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[St*3+1]+buf1[Wt*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[St*3+2]+buf1[Wt*3+2]))/4;
		}
		else if(mask[Nt_m]> 127.0f and mask[St_m]< 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt_m]> 127.0f and mask[St_m]< 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[Wt*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[Wt*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[Wt*3+2]))/4;
		}
		else if(mask[Nt_m]> 127.0f and mask[St_m]> 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[St*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[St*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[St*3+2]))/4;
		}
		else if(mask[Nt_m]< 127.0f and mask[St_m]< 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Et*3+2]))/4;
		}
		else if(mask[Nt_m]< 127.0f and mask[St_m]< 127.0f and mask[Wt_m]> 127.0f and mask[Et_m]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Wt*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Wt*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Wt*3+2]))/4;
		}
		else if(mask[Nt_m]< 127.0f and mask[St_m]> 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[St*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[St*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[St*3+2]))/4;
		}
		else if(mask[Nt_m]> 127.0f and mask[St_m]< 127.0f and mask[Wt_m]< 127.0f and mask[Et_m]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]))/4;
		}
		else{
			buf2[curt*3+0] = (fixed[curt*3+0])/4;
			buf2[curt*3+1] = (fixed[curt*3+1])/4;
			buf2[curt*3+2] = (fixed[curt*3+2])/4;
		}
	}
}




__global__ void shrink_a(const float *target,float *target_a,const float *mask,float *mask_a,const int wt,const int ht,const int wt_a,const int ht_a){
	const int yt_a = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt_a = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt_a = wt_a*yt_a+xt_a;
	if (yt_a < ht_a and xt_a < wt_a) {
		for(int row=0;row<8;row++){
			for(int col=0;col<8;col++){
				target_a[curt_a*3+0]+=target[(wt*(yt_a*8+row)+(xt_a*8+col))*3+0];
				target_a[curt_a*3+1]+=target[(wt*(yt_a*8+row)+(xt_a*8+col))*3+1];
				target_a[curt_a*3+2]+=target[(wt*(yt_a*8+row)+(xt_a*8+col))*3+2];
				mask_a[curt_a*3+0]+=mask[(wt*(yt_a*8+row)+(xt_a*8+col))*3+0];
				mask_a[curt_a*3+1]+=mask[(wt*(yt_a*8+row)+(xt_a*8+col))*3+1];
				mask_a[curt_a*3+2]+=mask[(wt*(yt_a*8+row)+(xt_a*8+col))*3+2];
			}
		}
		target_a[curt_a*3+0]/=64;
		target_a[curt_a*3+1]/=64;
		target_a[curt_a*3+2]/=64;
		mask_a[curt_a*3+0]/=64;
		mask_a[curt_a*3+1]/=64;
		mask_a[curt_a*3+2]/=64;
	}
}

__global__ void shrink_b(const float *target,float *target_a,const float *mask,float *mask_a,const int wt,const int ht,const int wt_a,const int ht_a){
	const int yt_a = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt_a = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt_a = wt_a*yt_a+xt_a;
	if (yt_a < ht_a and xt_a < wt_a) {
		for(int row=0;row<4;row++){
			for(int col=0;col<4;col++){
				target_a[curt_a*3+0]+=target[(wt*(yt_a*4+row)+(xt_a*4+col))*3+0];
				target_a[curt_a*3+1]+=target[(wt*(yt_a*4+row)+(xt_a*4+col))*3+1];
				target_a[curt_a*3+2]+=target[(wt*(yt_a*4+row)+(xt_a*4+col))*3+2];
				mask_a[curt_a*3+0]+=	mask[(wt*(yt_a*4+row)+(xt_a*4+col))*3+0];
				mask_a[curt_a*3+1]+=	mask[(wt*(yt_a*4+row)+(xt_a*4+col))*3+1];
				mask_a[curt_a*3+2]+=	mask[(wt*(yt_a*4+row)+(xt_a*4+col))*3+2];
			}
		}
		target_a[curt_a*3+0]/=16;
		target_a[curt_a*3+1]/=16;
		target_a[curt_a*3+2]/=16;
		  mask_a[curt_a*3+0]/=16;
		  mask_a[curt_a*3+1]/=16;
		  mask_a[curt_a*3+2]/=16;
	}
}

__global__ void shrink_c(const float *target,float *target_a,const float *mask,float *mask_a,const int wt,const int ht,const int wt_a,const int ht_a){
	const int yt_a = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt_a = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt_a = wt_a*yt_a+xt_a;
	if (yt_a < ht_a and xt_a < wt_a) {
		for(int row=0;row<2;row++){
			for(int col=0;col<2;col++){
				target_a[curt_a*3+0]+=target[(wt*(yt_a*2+row)+(xt_a*2+col))*3+0];
				target_a[curt_a*3+1]+=target[(wt*(yt_a*2+row)+(xt_a*2+col))*3+1];
				target_a[curt_a*3+2]+=target[(wt*(yt_a*2+row)+(xt_a*2+col))*3+2];
				mask_a[curt_a*3+0]+=	mask[(wt*(yt_a*2+row)+(xt_a*2+col))*3+0];
				mask_a[curt_a*3+1]+=	mask[(wt*(yt_a*2+row)+(xt_a*2+col))*3+1];
				mask_a[curt_a*3+2]+=	mask[(wt*(yt_a*2+row)+(xt_a*2+col))*3+2];
			}
		}
		target_a[curt_a*3+0]/=4;
		target_a[curt_a*3+1]/=4;
		target_a[curt_a*3+2]/=4;
		  mask_a[curt_a*3+0]/=4;
		  mask_a[curt_a*3+1]/=4;
		  mask_a[curt_a*3+2]/=4;
	}
}

__global__ void enlarge_a(const float *buf1_a, float *output_a,const int wt,const int ht,const int wt_a,const int ht_a){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt) {
		output_a[curt*3+0]=buf1_a[(wt_a*(yt/2)+xt/2)*3+0];
		output_a[curt*3+1]=buf1_a[(wt_a*(yt/2)+xt/2)*3+1];
		output_a[curt*3+2]=buf1_a[(wt_a*(yt/2)+xt/2)*3+2];
	}
}

__global__ void enlarge_b(const float *buf1_a, float *output_a,const int wt,const int ht,const int wt_a,const int ht_a){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt) {
		output_a[curt*3+0]=buf1_a[(wt_a*(yt/2)+xt/2)*3+0];
		output_a[curt*3+1]=buf1_a[(wt_a*(yt/2)+xt/2)*3+1];
		output_a[curt*3+2]=buf1_a[(wt_a*(yt/2)+xt/2)*3+2];
	}
}

__global__ void enlarge_c(const float *buf1_a, float *output_a,const int wt,const int ht,const int wt_a,const int ht_a){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt) {
		output_a[curt*3+0]=buf1_a[(wt_a*(yt/2)+xt/2)*3+0];
		output_a[curt*3+1]=buf1_a[(wt_a*(yt/2)+xt/2)*3+1];
		output_a[curt*3+2]=buf1_a[(wt_a*(yt/2)+xt/2)*3+2];
	}
}

__global__ void shrinkbackground_a(const float *background,float *background_a,const int wb,const int hb,const int wb_a,const int hb_a){
	const int yb_a = blockIdx.y * blockDim.y + threadIdx.y;
	const int xb_a = blockIdx.x * blockDim.x + threadIdx.x;
	const int curb_a = wb_a*yb_a+xb_a;
	if (yb_a < hb_a and xb_a < wb_a) {
		for(int row=0;row<8;row++){
			for(int col=0;col<8;col++){
				background_a[curb_a*3+0]+=background[(wb*(yb_a*8+row)+(xb_a*8+col))*3+0];
				background_a[curb_a*3+1]+=background[(wb*(yb_a*8+row)+(xb_a*8+col))*3+1];
				background_a[curb_a*3+2]+=background[(wb*(yb_a*8+row)+(xb_a*8+col))*3+2];
			}
		}
		background_a[curb_a*3+0]/=64;
		background_a[curb_a*3+1]/=64;
		background_a[curb_a*3+2]/=64;
	}
}

__global__ void shrinkbackground_b(const float *background,float *background_a,const int wb,const int hb,const int wb_a,const int hb_a){
	const int yb_a = blockIdx.y * blockDim.y + threadIdx.y;
	const int xb_a = blockIdx.x * blockDim.x + threadIdx.x;
	const int curb_a = wb_a*yb_a+xb_a;
	if (yb_a < hb_a and xb_a < wb_a) {
		for(int row=0;row<4;row++){
			for(int col=0;col<4;col++){
				background_a[curb_a*3+0]+=background[(wb*(yb_a*4+row)+(xb_a*4+col))*3+0];
				background_a[curb_a*3+1]+=background[(wb*(yb_a*4+row)+(xb_a*4+col))*3+1];
				background_a[curb_a*3+2]+=background[(wb*(yb_a*4+row)+(xb_a*4+col))*3+2];
			}
		}
		background_a[curb_a*3+0]/=16;
		background_a[curb_a*3+1]/=16;
		background_a[curb_a*3+2]/=16;
	}
}

__global__ void shrinkbackground_c(const float *background,float *background_a,const int wb,const int hb,const int wb_a,const int hb_a){
	const int yb_a = blockIdx.y * blockDim.y + threadIdx.y;
	const int xb_a = blockIdx.x * blockDim.x + threadIdx.x;
	const int curb_a = wb_a*yb_a+xb_a;
	if (yb_a < hb_a and xb_a < wb_a) {
		for(int row=0;row<2;row++){
			for(int col=0;col<2;col++){
				background_a[curb_a*3+0]+=background[(wb*(yb_a*2+row)+(xb_a*2+col))*3+0];
				background_a[curb_a*3+1]+=background[(wb*(yb_a*2+row)+(xb_a*2+col))*3+1];
				background_a[curb_a*3+2]+=background[(wb*(yb_a*2+row)+(xb_a*2+col))*3+2];
			}
		}
		background_a[curb_a*3+0]/=4;
		background_a[curb_a*3+1]/=4;
		background_a[curb_a*3+2]/=4;
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	) {
		
	int wt_a=wt/8;
	int ht_a=ht/8;
	int wb_a=wb/8;
	int hb_a=hb/8;
	int oy_a=oy/8;
	int ox_a=ox/8;
	int wt_b=wt/4;
	int ht_b=ht/4;
	int wb_b=wb/4;
	int hb_b=hb/4;
	int oy_b=oy/4;
	int ox_b=ox/4;
	int wt_c=wt/2;
	int ht_c=ht/2;
	int wb_c=wb/2;
	int hb_c=hb/2;
	int oy_c=oy/2;
	int ox_c=ox/2;
	// set up
	float *fixed, *buf1, *buf2;
	float *fixed_a, *buf1_a, *buf2_a, *background_a,*output_a,*mask_a,*target_a;
	float *fixed_b, *buf1_b, *buf2_b, *background_b,*output_b,*mask_b,*target_b;
	float *fixed_c, *buf1_c, *buf2_c, *background_c,*output_c,*mask_c,*target_c;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));
	cudaMalloc(&fixed_a, 		3*wt_a*ht_a*sizeof(float));
	cudaMalloc(&buf1_a,  		3*wt_a*ht_a*sizeof(float));
	cudaMalloc(&output_a,  		3*wt_b*ht_b*sizeof(float));
	cudaMalloc(&buf2_a,  		3*wt_a*ht_a*sizeof(float));
	cudaMalloc(&background_a, 	3*wb_a*hb_a*sizeof(float));
	cudaMalloc(&target_a, 		3*wt_a*ht_a*sizeof(float));
	cudaMalloc(&mask_a, 		3*wt_a*ht_a*sizeof(float));
	cudaMalloc(&fixed_b, 		3*wt_b*ht_b*sizeof(float));
	cudaMalloc(&buf1_b,  		3*wt_b*ht_b*sizeof(float));
	cudaMalloc(&output_b,  		3*wt_c*ht_c*sizeof(float));
	cudaMalloc(&buf2_b,  		3*wt_b*ht_b*sizeof(float));
	cudaMalloc(&background_b, 	3*wb_b*hb_b*sizeof(float));
	cudaMalloc(&target_b, 		3*wt_b*ht_b*sizeof(float));
	cudaMalloc(&mask_b, 		3*wt_b*ht_b*sizeof(float));
	cudaMalloc(&fixed_c, 		3*wt_c*ht_c*sizeof(float));
	cudaMalloc(&buf1_c,  		3*wt_c*ht_c*sizeof(float));
	cudaMalloc(&output_c,  		3*wt*ht*sizeof(float));
	cudaMalloc(&buf2_c,  		3*wt_c*ht_c*sizeof(float));
	cudaMalloc(&background_c, 	3*wb_c*hb_c*sizeof(float));
	cudaMalloc(&target_c, 		3*wt_c*ht_c*sizeof(float));
	cudaMalloc(&mask_c, 		3*wt_c*ht_c*sizeof(float));
	
	dim3 gdim_a(CeilDiv(wt_a,32), CeilDiv(ht_a,16)), bdim_a(32,16);
	dim3 gdim_back_a(CeilDiv(wb_a,32), CeilDiv(hb_a,16)), bdim_back_a(32,16);
	dim3 gdim_b(CeilDiv(wt_b,32), CeilDiv(ht_b,16)), bdim_b(32,16);
	dim3 gdim_back_b(CeilDiv(wb_b,32), CeilDiv(hb_b,16)), bdim_back_b(32,16);
	dim3 gdim_c(CeilDiv(wt_c,32), CeilDiv(ht_c,16)), bdim_c(32,16);
	dim3 gdim_back_c(CeilDiv(wb_c,32), CeilDiv(hb_c,16)), bdim_back_c(32,16);
	// initialize the iteration
	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	CalculateFixed<<<gdim, bdim>>>(
	background, target, mask, fixed,
	wb, hb, wt, ht, oy, ox
	);
//====================================	
	
	
	shrink_a<<<gdim_a,bdim_a>>>(target,target_a,mask,mask_a,wt,ht,wt_a,ht_a);
	shrinkbackground_a<<<gdim_back_a,bdim_back_a>>>(background,background_a,wb,hb,wb_a,hb_a);
	// initialize the iteration
	
	CalculateFixed
	<<<gdim_a, bdim_a>>>(
	background_a, target_a, mask_a, fixed_a,
	wb_a, hb_a, wt_a, ht_a, oy_a, ox_a
	);
	cudaMemcpy(buf1_a, target_a, sizeof(float)*3*wt_a*ht_a, cudaMemcpyDeviceToDevice);
	
	// iterate
	for (int i = 0; i <1000;i++){
		PoissonImageCloningIteration<<<gdim_a, bdim_a>>>(fixed_a, mask_a, buf1_a, buf2_a, wt_a, ht_a);
		PoissonImageCloningIteration<<<gdim_a, bdim_a>>>(fixed_a, mask_a, buf2_a, buf1_a, wt_a, ht_a);
	}
	enlarge_a<<<gdim_b,bdim_b>>>(buf1_a,output_a,wt_b,ht_b,wt_a,ht_a);
//================================_
	
	shrink_b<<<gdim_b,bdim_b>>>(target,target_b,mask,mask_b,wt,ht,wt_b,ht_b);
	shrinkbackground_b<<<gdim_back_b,bdim_back_b>>>(background,background_b,wb,hb,wb_b,hb_b);
	// initialize the iteration
	
	CalculateFixed
	<<<gdim_b, bdim_b>>>(
	background_b, target_b, mask_b, fixed_b,
	wb_b, hb_b, wt_b, ht_b, oy_b, ox_b
	);
	cudaMemcpy(buf1_b, output_a, sizeof(float)*3*wt_b*ht_b, cudaMemcpyDeviceToDevice);
	
	// iterate
	for (int i = 0; i <1000;i++){
		PoissonImageCloningIteration<<<gdim_b, bdim_b>>>(fixed_b, mask_b, buf1_b, buf2_b, wt_b, ht_b);
		PoissonImageCloningIteration<<<gdim_b, bdim_b>>>(fixed_b, mask_b, buf2_b, buf1_b, wt_b, ht_b);
	}
	enlarge_b<<<gdim_c,bdim_c>>>(buf1_b,output_b,wt_c,ht_c,wt_b,ht_b);
	

//=================================
	
	shrink_c<<<gdim_c,bdim_c>>>(target,target_c,mask,mask_c,wt,ht,wt_c,ht_c);
	shrinkbackground_c<<<gdim_back_c,bdim_back_c>>>(background,background_c,wb,hb,wb_c,hb_c);
	// initialize the iteration
	
	CalculateFixed
	<<<gdim_c, bdim_c>>>(
	background_c, target_c, mask_c, fixed_c,
	wb_c, hb_c, wt_c, ht_c, oy_c, ox_c
	);
	cudaMemcpy(buf1_c, output_b, sizeof(float)*3*wt_c*ht_c, cudaMemcpyDeviceToDevice);
	
	// iterate
	for (int i = 0; i <1000;i++){
		PoissonImageCloningIteration<<<gdim_c, bdim_c>>>(fixed_c, mask_c, buf1_c, buf2_c, wt_c, ht_c);
		PoissonImageCloningIteration<<<gdim_c, bdim_c>>>(fixed_c, mask_c, buf2_c, buf1_c, wt_c, ht_c);
	}
	enlarge_c<<<gdim,bdim>>>(buf1_c,output_c,wt,ht,wt_c,ht_c);
	
	
//=================================
	cudaMemset(buf1,0,sizeof(float)*3*wt*ht);
	cudaMemcpy(buf1, output_c, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
	for (int i = 0; i <1000;i++){
		PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, mask, buf1, buf2, wt, ht);
		PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, mask, buf2, buf1, wt, ht);
	}
	
	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	
	SimpleClone<<<gdim, bdim>>>(background, buf1, mask, output,wb, hb, wt, ht, oy, ox);
	
	
	
	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
	cudaFree(fixed_a);
	cudaFree(buf1_a);
	cudaFree(buf2_a);
	cudaFree(output_a);
	cudaFree(background_a);
	cudaFree(target_a);
	cudaFree(mask_a);
	
	cudaFree(fixed_b);
	cudaFree(buf1_b);
	cudaFree(buf2_b);
	cudaFree(output_b);
	cudaFree(background_b);
	cudaFree(target_b);
	cudaFree(mask_b);
	
	cudaFree(fixed_c);
	cudaFree(buf1_c);
	cudaFree(buf2_c);
	cudaFree(output_c);
	cudaFree(background_c);
	cudaFree(target_c);
	cudaFree(mask_c);
//=======================================	
	
	
	
	
	
	
	
}