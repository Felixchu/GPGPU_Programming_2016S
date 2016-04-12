#pragma once
#include <cstdint>
#include <memory>

using std::unique_ptr;

struct Lab2VideoInfo {
	unsigned w, h, n_frame;
	unsigned fps_n, fps_d;
};

class Lab2VideoGenerator {
	struct Impl;
	unique_ptr<Impl> impl;
public:
	Lab2VideoGenerator();
	~Lab2VideoGenerator();
	void get_info(Lab2VideoInfo &info);
	void Generate(uint8_t *yuv);
	void velocity();
	void project();
	void diffusion_h(float* , float* , float* , float* , float* , float* );
	void swap(float* ,float* ,float* ,float* );
	void set_vel_bound_h(float* , float* , float* , float* );
	void GaussSeidel_h(float* ,float* ,float* ,float* ,float* ,float* );
	void addvection_h(float* , float* , float* , float* ,float* ,float* ,float* , float* );
	void density();
	void diffusion_den_h(float* , float* , float* );
	void set_den_bound_h(float* , float* );
	void addvection_den_h(float* , float* , float* , float* ,float* );
	void action_h();
	float* d_vel_o_x;
	float* d_vel_o_y;
	float* d_vel_x	;
	float* d_vel_y	;
	float* d_den	;
	float* d_den_o	;
	float * d_vel_t_x;
	float* d_vel_t_y;
	float * d_vel_tt_x;
	float* d_vel_tt_y;
	float * d_den_t;
	int counter;
	float dt;
	float viscosity;
	int max_den;
};
