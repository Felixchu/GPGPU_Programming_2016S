#include "lab2.h"
#include <stdlib.h>
#include <iostream>
using namespace std;
#define d_vel_x(a,b) d_vel_x[(b)+(a)*blockDim.x*gridDim.x]
#define d_vel_y(a,b) d_vel_y[(b)+(a)*blockDim.x*gridDim.x]
#define d_vel_o_x(a,b) d_vel_o_x[(b)+(a)*blockDim.x*gridDim.x]
#define d_vel_o_y(a,b) d_vel_o_y[(b)+(a)*blockDim.x*gridDim.x]
#define d_vel_t_x(a,b) d_vel_t_x[(b)+(a)*blockDim.x*gridDim.x]
#define d_vel_t_y(a,b) d_vel_t_y[(b)+(a)*blockDim.x*gridDim.x]
#define d_den(a,b) d_den[(b)+(a)*blockDim.x*gridDim.x]
#define d_den_o(a,b) d_den_o[(b)+(a)*blockDim.x*gridDim.x]
#define gpudeclare() int x=threadIdx.x+blockIdx.x*blockDim.x;int y=threadIdx.y+blockIdx.y*blockDim.y;int id=x+y*blockDim.x*gridDim.x;
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME =500;


static const dim3 dimGrid(32,24);
static const dim3 dimBlock(20,20);
struct Lab2VideoGenerator::Impl {
	int t = 0;
};
__global__ void initial(float* d_vel_x,float* d_vel_y,float* d_vel_o_x,float* d_vel_o_y,float* d_vel_t_x,float* d_vel_t_y,float* d_den,float* d_den_o,float* d_den_t){
	int x=threadIdx.x+blockIdx.x*blockDim.x;
	int y=threadIdx.y+blockIdx.y*blockDim.y;
	int id=x+y*blockDim.x*gridDim.x;
	d_den[id]=x*800;
	d_vel_x[id]=0;
	d_vel_y[id]=0;
	d_den_o[id]=800;
	//d_den_o[id]=x*400;
	d_vel_o_x[id]=0;
	d_vel_o_y[id]=0;
	d_vel_t_x[id]=0;
	d_vel_t_y[id]=0;
	d_den_t[id]=0;
	//d_vel_o_x(1,x)=4;
	//d_vel_o_x(100,x)=2;
    //
	//	
	//d_vel_o_x(200,x)=-2;
	//
	if((x-100)*(x-100)+(y-100)*(y-100) <5000){
		d_vel_x[id]=y-100;
		d_vel_y[id]=x-100;
	}
	
	if((x-100)*(x-100)+(y-300)*(y-300) <5000){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-100;
	}
	
	if((x-540)*(x-540)+(y-100)*(y-100) <5000){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-540;
	}
	if((x-540)*(x-540)+(y-300)*(y-300) <5000){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-540;
	}
	
	if((x-200)*(x-200)+(y-100)*(y-100) <5000){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-540;
	}
	if((x-300)*(x-300)+(y-300)*(y-300) <5000){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-540;
	}
	if((x-400)*(x-400)+(y-100)*(y-100) <5000){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-540;
	}
	if((x-500)*(x-500)+(y-300)*(y-300) <5000){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-540;
	}
	
}
__global__ void read(uint8_t* yuv,float* d_den,int max_den){
	gpudeclare();
	yuv[id]=(d_den[id]<max_den)?255*d_den[id]/max_den:255;
	//yuv[id]=d_den[id];
}

__global__ void add(float* d_vel_x,float* d_vel_y,float* d_vel_o_x,float* d_vel_o_y,int dt){
	int x=threadIdx.x+blockIdx.x*blockDim.x;
	int y=threadIdx.y+blockIdx.y*blockDim.y;
	int id=x+y*blockDim.x*gridDim.x;
	d_vel_x[id]+=d_vel_o_x[id]*dt;
	d_vel_y[id]+=d_vel_o_y[id]*dt;
}

__global__ void add_den(float* d_den,float* d_den_o,float dt){
	gpudeclare();
	d_den[id]+=d_den_o[id]*dt;
}

__global__ void diffusion(float* d_vel_x,float* d_vel_y,float* d_vel_o_x,float* d_vel_o_y,float*d_vel_t_x ,float* d_vel_t_y,float dt,float viscosity){
	gpudeclare();
	float a= dt*viscosity*W*H;
		if((x>0) && (x<W-1) && (y>=1) && (y<H-1)){
			d_vel_x[id]=(d_vel_o_x[id]+a*(d_vel_o_x(y-1,x)+d_vel_o_x(y+1,x)+d_vel_o_x(y,x-1)+d_vel_o_x(y,x+1))-4*d_vel_o_x(y,x));
			d_vel_y[id]=(d_vel_o_y[id]+a*(d_vel_o_y(y-1,x)+d_vel_o_y(y+1,x)+d_vel_o_y(y,x-1)+d_vel_o_y(y,x+1))-4*d_vel_o_y(y,x));
		}
}

__global__ void diffusion_den(float* d_den,float* d_den_o,float* d_den_t,float dt,float viscosity){
	gpudeclare();
	float a= dt*viscosity*W*H;
		if((x>0) && (x<W-1) && (y>=1) && (y<H-1)){
			d_den[id]=(d_den_o[id]+a*(d_den_o(y-1,x)+d_den_o(y+1,x)+d_den_o(y,x-1)+d_den_o(y,x+1)-4*d_den_o(y,x)));
		}
}

__global__ void set_vel_bound(float* d_vel_x,float* d_vel_y,float* d_vel_t_x,float* d_vel_t_y){
	gpudeclare();
	d_vel_t_x[id]=d_vel_x[id];
	d_vel_t_y[id]=d_vel_y[id];
	if(x==0)
		d_vel_t_y[id]= -d_vel_y(y,1);
	if(x==W-1)
		d_vel_t_y[id]= -d_vel_y(y,W-2);
	if(y==0)
		d_vel_t_x[id]= -d_vel_x(1,x);
	if(y==H-1)
		d_vel_t_x[id]= -d_vel_x(H-2,x);
	if(x==0 && y==0){
		d_vel_t_x[id]=0.5 *(d_vel_x(1,0)+d_vel_x(0,1));
		d_vel_t_y[id]=0.5 *(d_vel_y(1,0)+d_vel_y(0,1));
	}
	if(x==W-1 && y==0){
		d_vel_t_x[id]=0.5 *(d_vel_x(1,W-1)+d_vel_x(0,W-1));
		d_vel_t_y[id]=0.5 *(d_vel_y(1,W-1)+d_vel_y(0,W-1));
	}
	if(x==0 && y==H-1){
		d_vel_t_x[id]=0.5 *(d_vel_x(H-2,0)+d_vel_x(H-1,1));
		d_vel_t_y[id]=0.5 *(d_vel_y(H-2,0)+d_vel_y(H-1,1));
	}
	if(x==W-1 && y==H-1){
		d_vel_t_x[id]= 0.5*(d_vel_x(H-2,W-1)+d_vel_x(H-1,W-2));
		d_vel_t_y[id]= 0.5*(d_vel_y(H-2,W-1)+d_vel_y(H-1,W-2));
	}

	if(y>200 && y<300 && x>200&& x<300){
		d_vel_t_x[id]= -d_vel_x[id];
		d_vel_t_y[id]= -d_vel_y[id];
	}
}
__global__ void set_den_bound(float* d_den,float* d_den_t){
	gpudeclare();
	d_den_t[id]=d_den[id];
	if(x==0)
		d_den_t[id]= -d_den(y,1);
	if(x==W-1)
		d_den_t[id]= -d_den(y,W-2);
	if(y==0)
		d_den_t[id]= -d_den(1,x);
	if(y==H-1)
		d_den_t[id]= -d_den(H-2,x);
	if(x==0 && y==0){
		d_den_t[id]=0.5 *(d_den(1,0)+d_den(0,1));
	}
	if(x==W-1 && y==0){
		d_den_t[id]=0.5 *(d_den(1,W-1)+d_den(0,W-1));
	}
	if(x==0 && y==H-1){
		d_den_t[id]=0.5 *(d_den(H-2,0)+d_den(H-1,1));
	}
	if(x==W-1 && y==H-1){
		d_den_t[id]= 0.5*(d_den(H-2,W-1)+d_den(H-1,W-2));
	}
}


__global__ void GaussSeidel(float* d_vel_x,float* d_vel_y,float*d_vel_o_x ,float* d_vel_o_y,float*d_vel_t_x ,float* d_vel_t_y){
	gpudeclare();
	if((x>0) && (x<W-1) && (y>=1) && (y<H-1)){
		d_vel_o_y[id]= (d_vel_x(y,x) + d_vel_y(y-1,x)+d_vel_y(y+1,x)+d_vel_y(y,x-1)+d_vel_y(y,x+1)-4*d_vel_y(y,x)); 
	}
}

__global__ void project_set_vel(float* d_vel_x,float* d_vel_y,float*d_vel_o_x ,float* d_vel_o_y){
	gpudeclare();
	float h_x=1.0/W;
	float h_y=1.0/H;
	if((x>0) && (x<W-1) && (y>=1) && (y<H-1)){
		d_vel_x[id]=d_vel_x[id]-0.5*(d_vel_o_y(y+1,x)-d_vel_o_y(y-1,x))/h_x;
		d_vel_y[id]=d_vel_y[id]-0.5*(d_vel_o_y(y+1,x)-d_vel_o_y(y-1,x))/h_y;
	}
}

__global__ void project_set_div(float* d_vel_x,float* d_vel_y,float*d_vel_o_x ,float* d_vel_o_y){
	gpudeclare();
	float h_x=1.0/W;
	if((x>0) && (x<W-1) && (y>=1) && (y<H-1)){
		d_vel_o_x[id]=-0.5 * h_x *(d_vel_x(y+1,x)-d_vel_x(y-1,x)+d_vel_y(y,x+1)-d_vel_y(y,x-1));
		d_vel_o_y[id]=0;
	}
}

__global__ void addvection(float* d_vel_x,float*d_vel_y,float*d_vel_o_x,float*d_vel_o_y,float*d_vel_in_x,float*d_vel_in_y,float dt){
	gpudeclare();
	if((x>0) && (x<W-1) && (y>=1) && (y<H-1)){
		float x_i= x-dt*W*d_vel_in_x[id];
		float y_i= y-dt*H*d_vel_in_y[id];
		if(x_i<0.5) x_i=0.5;
		if(x_i>W+0.5) x_i=W+0.5;
		int x_i0=(int) x_i;
		int x_i1=x_i0+1;
		if(y_i<0.5) y_i=0.5;
		if(y_i>W+0.5) y_i=W+0.5;
		int y_i0=(int) y_i;
		int y_i1=y_i0+1;
		float x_i_f = x - x_i0;
		float x_i_fc = 1 - x_i_f;
		float y_i_f = y - y_i0;
		float y_i_fc = 1 - y_i_f;
		
		d_vel_x[id]=x_i_fc *(y_i_fc*d_vel_o_x(y_i0,x_i0)+y_i_f*d_vel_o_x(y_i0,x_i1))+x_i_f*(y_i_fc*d_vel_o_x(y_i1,x_i0)+y_i_f*d_vel_o_x(y_i1,x_i1));
		d_vel_y[id]=x_i_fc *(y_i_fc*d_vel_o_y(y_i0,x_i0)+y_i_f*d_vel_o_y(y_i0,x_i1))+x_i_f*(y_i_fc*d_vel_o_y(y_i1,x_i0)+y_i_f*d_vel_o_y(y_i1,x_i1));
	}
}


__global__ void addvection_den(float* d_den,float*d_den_o,float*d_den_t,float*d_vel_x,float*d_vel_y,float dt){
	gpudeclare();
	if((x>0) && (x<W-1) && (y>=1) && (y<H-1)){
		float x_i= x-dt*W*d_vel_x[id];
		float y_i= y-dt*H*d_vel_y[id];
		if(x_i<0.5) x_i=0.5;
		if(x_i>W+0.5) x_i=W+0.5;
		int x_i0=(int) x_i;
		int x_i1=x_i0+1;
		if(y_i<0.5) y_i=0.5;
		if(y_i>W+0.5) y_i=W+0.5;
		int y_i0=(int) y_i;
		int y_i1=y_i0+1;
		float x_i_f = x - x_i0;
		float x_i_fc = 1 - x_i_f;
		float y_i_f = y - y_i0;
		float y_i_fc = 1 - y_i_f;
		
		d_den[id]=x_i_fc *(y_i_fc*d_den_o(y_i0,x_i0)+y_i_f*d_den_o(y_i0,x_i1))+x_i_f*(y_i_fc*d_den_o(y_i1,x_i0)+y_i_f*d_den_o(y_i1,x_i1));
	}
}

__global__ void action_1(float* d_vel_x,float* d_vel_y,float* d_den,float* d_vel_o_x,float* d_vel_o_y,float* d_den_o){
	gpudeclare();

	d_vel_x[id]=0;
	d_vel_y[id]=0;
	d_vel_o_x[id]=0;
	d_vel_o_y[id]=0;
	if((x-200)*(x-200)+(y-100)*(y-100) <2000){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-100;
	}
	
	if((x-200)*(x-200)+(y-300)*(y-300) <2000){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-100;
	}
	
	if((x-540)*(x-540)+(y-100)*(y-100) <2000){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-540;
	}
	if((x-540)*(x-540)+(y-300)*(y-300) <2000){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-540;
	}

}

__global__ void action_2(float* d_vel_x,float* d_vel_y,float* d_den,float* d_vel_o_x,float* d_vel_o_y,float* d_den_o){
	gpudeclare();

	d_vel_x[id]=0;
	d_vel_y[id]=0;
	d_vel_o_x[id]=0;
	d_vel_o_y[id]=0;
	if((x-200)*(x-200)+(y-100)*(y-100) <5000){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-100;
	}
	
	if((x-200)*(x-200)+(y-300)*(y-300) <5000){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-100;
	}
	
	if((x-540)*(x-540)+(y-100)*(y-100) <5000){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-540;
	}
	if((x-540)*(x-540)+(y-300)*(y-300) <5000){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-540;
	}

}

__global__ void action_3(float* d_vel_x,float* d_vel_y,float* d_den,float* d_vel_o_x,float* d_vel_o_y,float* d_den_o){
	gpudeclare();
	d_vel_x[id]=0;
	d_vel_y[id]=0;
	d_vel_o_x[id]=0;
	d_vel_o_y[id]=0;
	if((x-200)*(x-200)+(y-100)*(y-100) <200){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-100;
	}
	
	if((x-200)*(x-200)+(y-300)*(y-300) <200){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-100;
	}
	
	if((x-540)*(x-540)+(y-100)*(y-100) <200){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-540;
	}
	if((x-540)*(x-540)+(y-300)*(y-300) <200){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-540;
	}
	if((x-400)*(x-400)+(y-100)*(y-100) <200){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-400;
	}
	
	if((x-400)*(x-400)+(y-300)*(y-300) <200){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-400;
	}
	
	if((x-540)*(x-540)+(y-200)*(y-200) <200){
		d_vel_o_x[id]=y-200;
		d_vel_o_y[id]=x-540;
	}
	if((x-400)*(x-400)+(y-200)*(y-200) <200){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-400;
	}
}
__global__ void action_4(float* d_vel_x,float* d_vel_y,float* d_den,float* d_vel_o_x,float* d_vel_o_y,float* d_den_o){
	gpudeclare();

	d_vel_x[id]=0;
	d_vel_y[id]=0;
	d_vel_o_x[id]=0;
	d_vel_o_y[id]=0;
	if((x-300)*(x-300)+(y-200)*(y-200) <1000){
		d_vel_o_x[id]=y-200;
		d_vel_o_y[id]=x-300;
	}
	
	if((x-300)*(x-300)+(y-200)*(y-200) <5000){
		d_vel_o_x[id]=y-200;
		d_vel_o_y[id]=x-300;
	}
	
	if((x-300)*(x-300)+(y-200)*(y-200) <10000){
		d_vel_o_x[id]=y-200;
		d_vel_o_y[id]=x-300;
	}
	if((x-300)*(x-300)+(y-200)*(y-200) <20000){
		d_vel_o_x[id]=y-200;
		d_vel_o_y[id]=x-300;
	}
	
}


__global__ void action_5(float* d_vel_x,float* d_vel_y,float* d_den,float* d_vel_o_x,float* d_vel_o_y,float* d_den_o){
	gpudeclare();

	d_vel_x[id]=0;
	d_vel_y[id]=0;
	d_vel_o_x[id]=0;
	d_vel_o_y[id]=0;
	if((x-200)*(x-200)+(y-100)*(y-100) <2000){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-100;
	}
	
	if((x-200)*(x-200)+(y-300)*(y-300) <2000){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-100;
	}
	
	if((x-540)*(x-540)+(y-100)*(y-100) <2000){
		d_vel_o_x[id]=y-100;
		d_vel_o_y[id]=x-540;
	}
	if((x-540)*(x-540)+(y-300)*(y-300) <2000){
		d_vel_o_x[id]=y-300;
		d_vel_o_y[id]=x-540;
	}
}


__global__ void action_6(float* d_vel_x,float* d_vel_y,float* d_den,float* d_vel_o_x,float* d_vel_o_y,float* d_den_o){
	gpudeclare();

	d_vel_x[id]=0;
	d_vel_y[id]=0;
	d_vel_o_x[id]=0;
	d_vel_o_y[id]=0;
	if((x<300) && (x>100) && (y<300) && (y>100)){
		d_vel_o_x[id]=100;
		d_vel_o_y[id]=100;
	}
	if((x<400) && (x>300) && (y<300) && (y>100)){
		d_vel_o_x[id]=200;
		d_vel_o_y[id]=200;
	}
	if((x<500) && (x>450) && (y<300) && (y>100)){
		d_vel_o_x[id]=300;
		d_vel_o_y[id]=300;
	}
	if((x<600) && (x>550) && (y<300) && (y>100)){
		d_vel_o_x[id]=400;
		d_vel_o_y[id]=400;
	}
	if((x<300) && (x>100) && (y<400) && (y>350)){
		d_vel_o_x[id]=500;
		d_vel_o_y[id]=500;
	}
	if((x<300) && (x>100) && (y<100) && (y>20)){
		d_vel_o_x[id]=600;
		d_vel_o_y[id]=600;
	}
	
	

}

__global__ void action_7(float* d_vel_x,float* d_vel_y,float* d_den,float* d_vel_o_x,float* d_vel_o_y,float* d_den_o){
	gpudeclare();

	d_vel_x[id]=0;
	d_vel_y[id]=0;
	d_vel_o_x[id]=0;
	d_vel_o_y[id]=0;
	if((x<300) && (x>100) && (y<300) && (y>100)){
		d_vel_o_x[id]=100;
		d_vel_o_y[id]=100;
	}
}



void  Lab2VideoGenerator::swap(float* d_vel_x,float* d_vel_y,float* d_vel_o_x,float* d_vel_o_y){
	float * temp;
	temp=d_vel_x;
	d_vel_x=d_vel_o_x;
	d_vel_o_x=temp;
	temp=d_vel_y;
	d_vel_y=d_vel_o_y;
	d_vel_o_y=temp;
}


void  Lab2VideoGenerator::set_vel_bound_h(float* d_vel_x, float* d_vel_y, float* d_vel_t_x, float* d_vel_t_y){
	cudaMemset(d_vel_t_x, 0, W*H);
	cudaMemset(d_vel_t_y, 0, W*H);
	set_vel_bound<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_vel_t_x,d_vel_t_y);
	cudaDeviceSynchronize();
	swap(d_vel_x,d_vel_y,d_vel_t_x,d_vel_t_y);
}

void  Lab2VideoGenerator::set_den_bound_h(float* d_den, float* d_den_t){
	cudaMemset(d_den_t, 0, W*H);
	set_den_bound<<<dimGrid,dimBlock>>>(d_den,d_den_t);
	cudaDeviceSynchronize();
	swap(d_den,NULL,d_den_t,NULL);
}

void  Lab2VideoGenerator::GaussSeidel_h(float* d_vel_x,float* d_vel_y,float* d_vel_o_x,float* d_vel_o_y,float* d_vel_t_x,float* d_vel_t_y){
		GaussSeidel<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y,d_vel_t_x,d_vel_t_y);
		cudaDeviceSynchronize();
		swap(NULL,d_vel_t_y,NULL,d_vel_o_y);
		set_vel_bound_h(d_vel_o_x,d_vel_o_y,d_vel_t_x,d_vel_t_y);
		set_vel_bound_h(d_vel_o_x,d_vel_o_y,d_vel_t_x,d_vel_t_y);
}

void Lab2VideoGenerator::project(){
	project_set_div<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y);
	cudaDeviceSynchronize();
	set_vel_bound_h(d_vel_o_x,d_vel_o_y,d_vel_t_x,d_vel_t_y);
	GaussSeidel_h(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y,d_vel_t_x,d_vel_t_y);
	project_set_vel<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y);
	cudaDeviceSynchronize();
	set_vel_bound_h(d_vel_x,d_vel_y,d_vel_t_x,d_vel_t_y);
}

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
	cudaMalloc((void**)&d_vel_x,H*W*sizeof(float));
	cudaMalloc((void**)&d_vel_y,H*W*sizeof(float));
	cudaMalloc((void**)&d_den,H*W*sizeof(float));
	cudaMalloc((void**)&d_vel_o_x,H*W*sizeof(float));
	cudaMalloc((void**)&d_vel_o_y,H*W*sizeof(float));
	cudaMalloc((void**)&d_den_o,H*W*sizeof(float));
	cudaMalloc((void**)&d_vel_t_x,H*W*sizeof(float));
	cudaMalloc((void**)&d_vel_t_y,H*W*sizeof(float));
	cudaMalloc((void**)&d_vel_tt_x,H*W*sizeof(float));
	cudaMalloc((void**)&d_vel_tt_y,H*W*sizeof(float));
	cudaMalloc((void**)&d_den_t,H*W*sizeof(float));
	initial<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y,d_vel_t_x,d_vel_t_y,d_den,d_den_o,d_den_t);
	cudaDeviceSynchronize();
	dt=0.01;
	viscosity=0.02;
	counter=0;
	max_den=1000;
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};




void Lab2VideoGenerator::diffusion_h(float* d_vel_x, float* d_vel_y, float* d_vel_o_x, float* d_vel_o_y, float* d_vel_t_x, float* d_vel_t_y){
		diffusion<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y,d_vel_t_x,d_vel_t_y,dt,viscosity);
		cudaDeviceSynchronize();
	set_vel_bound_h(d_vel_x,d_vel_y,d_vel_t_x,d_vel_t_y);
}

void Lab2VideoGenerator::diffusion_den_h(float* d_den, float* d_den_o, float* d_den_t){
		diffusion_den<<<dimGrid,dimBlock>>>(d_den,d_den_o,d_den_t,dt,viscosity);
		cudaDeviceSynchronize();
	set_den_bound_h(d_den,d_den_t);
}

void Lab2VideoGenerator::addvection_h(float* d_vel_x, float* d_vel_y, float* d_vel_o_x, float* d_vel_o_y,float* d_vel_in_x,float* d_vel_in_y,float* d_vel_t_x, float* d_vel_t_y){
	addvection<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y,d_vel_in_x,d_vel_in_y,dt);
	cudaDeviceSynchronize();
	set_vel_bound_h(d_vel_x,d_vel_y,d_vel_t_x,d_vel_t_y);
}
void Lab2VideoGenerator::addvection_den_h(float* d_den, float* d_den_o, float* d_den_t, float* d_vel_x,float* d_vel_y){
	addvection_den<<<dimGrid,dimBlock>>>(d_den,d_den_o,d_den_t,d_vel_x,d_vel_y,dt);
	cudaDeviceSynchronize();
	set_den_bound_h(d_den,d_den_t);
}

void Lab2VideoGenerator::velocity(){
	add<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y,dt);
	cudaDeviceSynchronize();
	swap(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y);
	diffusion_h(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y,d_vel_t_x,d_vel_t_y);
	project();
	swap(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y);
	addvection_h(d_vel_x,d_vel_y,d_vel_o_x,d_vel_o_y,d_vel_o_x,d_vel_o_y,d_vel_t_x,d_vel_t_y);
	project();
}

void Lab2VideoGenerator::density(){
	add_den<<<dimGrid,dimBlock>>>(d_den,d_den_o,dt);
	swap(d_den,NULL,d_den_o,NULL);
	diffusion_den_h(d_den,d_den_o,d_den_t);
	swap(d_den,NULL,d_den_o,NULL);
	addvection_den_h(d_den,d_den_o,d_den_t,d_vel_x,d_vel_y);
}

void Lab2VideoGenerator::action_h(){
	if(counter==48)
		action_1<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_den,d_vel_o_x,d_vel_o_y,d_den_o);
	if(counter==96)
		action_2<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_den,d_vel_o_x,d_vel_o_y,d_den_o);
	if(counter==144){
		action_3<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_den,d_vel_o_x,d_vel_o_y,d_den_o);
	}
		
	if(counter==192){
		
		action_4<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_den,d_vel_o_x,d_vel_o_y,d_den_o);
		
	}
	if(counter==48*5)
		action_5<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_den,d_vel_o_x,d_vel_o_y,d_den_o);
	if(counter==48*6)
		action_6<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_den,d_vel_o_x,d_vel_o_y,d_den_o);
	if(counter==48*7)
		action_7<<<dimGrid,dimBlock>>>(d_vel_x,d_vel_y,d_den,d_vel_o_x,d_vel_o_y,d_den_o);

	counter++;
		cout<<"HI";
}

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
	velocity();
	density();
	cudaDeviceSynchronize();
	read<<<dimGrid,dimBlock>>>(yuv,d_den,max_den);
	cudaDeviceSynchronize();
	cudaMemset(yuv+W*H, 128, W*H/2);
	action_h();
	cout<<counter<<" ";
	
}
