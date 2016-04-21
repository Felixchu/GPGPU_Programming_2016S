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
	int Nt=curt-wt;
	int St=curt+wt;
	int Wt=curt-1;
	int Et=curt+1;
	if (yt < ht and xt < wt and mask[curt] > 127.0f and yt>1 and xt>1) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		int Nb=curb-wb;
		int Sb=curb+wb;
		int Eb=curb+1;
		int Wb=curb+1;
		if (1 <= yb and yb < hb-1 and 1 <= xb and xb < wb-1) {
			if(mask[Nt]> 127.0f and mask[St]> 127.0f and mask[Wt]> 127.0f and mask[Et]> 127.0f){
				fixed[curt*3+0] = 4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0]);
				fixed[curt*3+1] = 4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1]);
				fixed[curt*3+2] = 4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2]);
				
			}
			else if(mask[Nt]< 127.0f and mask[St]> 127.0f and mask[Wt]> 127.0f and mask[Et]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2];
			}
			else if(mask[Nt]> 127.0f and mask[St]< 127.0f and mask[Wt]> 127.0f and mask[Et]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Sb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Sb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Sb*3+2];
			}
			else if(mask[Nt]> 127.0f and mask[St]> 127.0f and mask[Wt]< 127.0f and mask[Et]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Wb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Wb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Wb*3+2];
			}
			else if(mask[Nt]> 127.0f and mask[St]> 127.0f and mask[Wt]> 127.0f and mask[Et]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Eb*3+2];
			}
			else if(mask[Nt]< 127.0f and mask[St]< 127.0f and mask[Wt]> 127.0f and mask[Et]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Sb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Sb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Sb*3+2];
			}
			else if(mask[Nt]< 127.0f and mask[St]> 127.0f and mask[Wt]< 127.0f and mask[Et]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Wb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Wb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Wb*3+2];
			}
			else if(mask[Nt]< 127.0f and mask[St]> 127.0f and mask[Wt]> 127.0f and mask[Et]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Eb*3+2];
			}
			else if(mask[Nt]> 127.0f and mask[St]< 127.0f and mask[Wt]< 127.0f and mask[Et]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Sb*3+0]+background[Wb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Sb*3+1]+background[Wb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Sb*3+2]+background[Wb*3+2];
			}
			else if(mask[Nt]> 127.0f and mask[St]< 127.0f and mask[Wt]> 127.0f and mask[Et]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Sb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Sb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Sb*3+2]+background[Eb*3+2];
			}
			else if(mask[Nt]> 127.0f and mask[St]> 127.0f and mask[Wt]< 127.0f and mask[Et]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Wb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Wb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Wb*3+2]+background[Eb*3+2];
			}
			else if(mask[Nt]< 127.0f and mask[St]< 127.0f and mask[Wt]< 127.0f and mask[Et]> 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Sb*3+0]+background[Wb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Sb*3+1]+background[Wb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Sb*3+2]+background[Wb*3+2];
			}
			else if(mask[Nt]< 127.0f and mask[St]< 127.0f and mask[Wt]> 127.0f and mask[Et]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Sb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Sb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Sb*3+2]+background[Eb*3+2];
			}
			else if(mask[Nt]< 127.0f and mask[St]> 127.0f and mask[Wt]< 127.0f and mask[Et]< 127.0f){
				fixed[curt*3+0] =4*target[curt*3+0]-(target[Nt*3+0]+target[St*3+0]+target[Wt*3+0]+target[Et*3+0])+background[Nb*3+0]+background[Wb*3+0]+background[Eb*3+0];
				fixed[curt*3+1] =4*target[curt*3+1]-(target[Nt*3+1]+target[St*3+1]+target[Wt*3+1]+target[Et*3+1])+background[Nb*3+1]+background[Wb*3+1]+background[Eb*3+1];
				fixed[curt*3+2] =4*target[curt*3+2]-(target[Nt*3+2]+target[St*3+2]+target[Wt*3+2]+target[Et*3+2])+background[Nb*3+2]+background[Wb*3+2]+background[Eb*3+2];
			}
			else if(mask[Nt]> 127.0f and mask[St]< 127.0f and mask[Wt]< 127.0f and mask[Et]< 127.0f){
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
	int Nt=curt-wt;
	int St=curt+wt;
	int Wt=curt-1;
	int Et=curt+1;
	if (yt < ht and xt < wt and mask[curt] > 127.0f and yt>1 and xt>1) {
		if(mask[Nt]> 127.0f and mask[St]> 127.0f and mask[Wt]> 127.0f and mask[Et]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[St*3+0]+buf1[Wt*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[St*3+1]+buf1[Wt*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[St*3+2]+buf1[Wt*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt]< 127.0f and mask[St]> 127.0f and mask[Wt]> 127.0f and mask[Et]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[St*3+0]+buf1[Wt*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[St*3+1]+buf1[Wt*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[St*3+2]+buf1[Wt*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt]> 127.0f and mask[St]< 127.0f and mask[Wt]> 127.0f and mask[Et]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[Wt*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[Wt*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[Wt*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt]> 127.0f and mask[St]> 127.0f and mask[Wt]< 127.0f and mask[Et]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[St*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[St*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[St*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt]> 127.0f and mask[St]> 127.0f and mask[Wt]> 127.0f and mask[Et]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[St*3+0]+buf1[Wt*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[St*3+1]+buf1[Wt*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[St*3+2]+buf1[Wt*3+2]))/4;
		}
		else if(mask[Nt]< 127.0f and mask[St]< 127.0f and mask[Wt]> 127.0f and mask[Et]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Wt*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Wt*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Wt*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt]< 127.0f and mask[St]> 127.0f and mask[Wt]< 127.0f and mask[Et]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(+buf1[St*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(+buf1[St*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(+buf1[St*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt]< 127.0f and mask[St]> 127.0f and mask[Wt]> 127.0f and mask[Et]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[St*3+0]+buf1[Wt*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[St*3+1]+buf1[Wt*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[St*3+2]+buf1[Wt*3+2]))/4;
		}
		else if(mask[Nt]> 127.0f and mask[St]< 127.0f and mask[Wt]< 127.0f and mask[Et]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[Et*3+2]))/4;
		}
		else if(mask[Nt]> 127.0f and mask[St]< 127.0f and mask[Wt]> 127.0f and mask[Et]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[Wt*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[Wt*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[Wt*3+2]))/4;
		}
		else if(mask[Nt]> 127.0f and mask[St]> 127.0f and mask[Wt]< 127.0f and mask[Et]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Nt*3+0]+buf1[St*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Nt*3+1]+buf1[St*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Nt*3+2]+buf1[St*3+2]))/4;
		}
		else if(mask[Nt]< 127.0f and mask[St]< 127.0f and mask[Wt]< 127.0f and mask[Et]> 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Et*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Et*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Et*3+2]))/4;
		}
		else if(mask[Nt]< 127.0f and mask[St]< 127.0f and mask[Wt]> 127.0f and mask[Et]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[Wt*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[Wt*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[Wt*3+2]))/4;
		}
		else if(mask[Nt]< 127.0f and mask[St]> 127.0f and mask[Wt]< 127.0f and mask[Et]< 127.0f){
			buf2[curt*3+0] = (fixed[curt*3+0]+(buf1[St*3+0]))/4;
			buf2[curt*3+1] = (fixed[curt*3+1]+(buf1[St*3+1]))/4;
			buf2[curt*3+2] = (fixed[curt*3+2]+(buf1[St*3+2]))/4;
		}
		else if(mask[Nt]> 127.0f and mask[St]< 127.0f and mask[Wt]< 127.0f and mask[Et]< 127.0f){
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


void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	) {
	// set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));
	
	// initialize the iteration
	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	CalculateFixed<<<gdim, bdim>>>(
	background, target, mask, fixed,
	wb, hb, wt, ht, oy, ox
	);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
	
	// iterate
	for (int i = 0; i < 10; ++i) {
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
}