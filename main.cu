#include "stdlib.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"

#define MAX_THREADS (512)
#define PI (3.14159265)

__device__  float kernel_sinm(float fn, float fce, float t)
{
	return sin((fn - fce)*t);
	
} 

__device__  float kernel_sinp(float fn, float fce, float t)
{
	return sin((fn + fce)*t);
}

__device__  float kernel_cosm(float fn, float fce, float t)
{
	return cos((fn - fce)*t);
	
} 

__device__  float kernel_cosp(float fn, float fce, float t)
{
	return cos((fn + fce)*t);
}

__global__ void global_calculate_vt(float* fn, float* fce, float* vnp, float* vnm, size_t nh, float* tn, float* vt)
{      
	size_t nt = blockDim.x*blockIdx.x + threadIdx.x;
	float t = tn[nt];
	size_t i;
	float kfce = *fce;
	float sinp;
	float sinm;
	
	//load vn coeff to shared memory
	float vnt = 0;
	for ( i = 0; i < nh; ++i )
	{
		float fnt = fn[i];
		float vnpt = vnp[i];
		float vnmt = vnm[i];
		sinp = kernel_sinp(fnt, kfce, t);
		sinm = kernel_sinm(fnt, kfce, t);
		vnt  += vnpt*sinp + vnmt*sinm;
	}
	vt[nt] = vnt;
}

__global__ void global_calculate_et(float* fn, float* fce, float* enp, float* enm, size_t nh, float* tn, float* et)
{      
	size_t nt = blockDim.x*blockIdx.x + threadIdx.x;
	float t = tn[nt];
	size_t i;
	float kfce = *fce;
	float cosp;
	float cosm;
	
	//load vn coeff to shared memory
	float ent = 0;
	for ( i = 0; i < nh; ++i )
	{
		float fnt = fn[i];
		float enpt = enp[i];
		float enmt = enm[i];
		cosp = kernel_cosp(fnt, kfce, t);
		cosm = kernel_cosm(fnt, kfce, t);
		ent  += enpt*cosp + enmt*cosm;
	}
	et[nt] = ent;
}

void generate_an(float a0, size_t nh, float* an)
{
	size_t i;
	float a = a0*4/PI;
	for (i = 0; i < nh; ++i)
	{
		an[i] = a/(2*i+1);
	}
}

void generate_fn(float fm, size_t nh, float* fn)
{
	size_t i;
	for (i = 0; i < nh; ++i)
	{
		fn[i] = fm*(2*i+1);
	}
}

void generate_vn(float* an, float* fn, float fce,
		 size_t nh, float* vnp, float* vnm)
{
	size_t i;
	float fce2 = fce*fce;
	for (i = 0; i < nh; ++i)
	{
		float fs = fn[i] + fce;
		float fd = fn[i] - fce;
     
		vnp[i] = 0.5*an[i]*(fs/(fce2 - pow(fs,2)));
		vnm[i] = -0.5*an[i]*(fd/(fce2 - pow(fd,2)));
	}	
}

void generate_en(float* an, float* fn, float fce,
		 size_t nh, float* enp, float* enm)
{
	size_t i;
	for (i = 0; i < nh; ++i)
	{     
		enm[i] = 0.5*an[i];
		enp[i] = -0.5*an[i];
	}	
}

void generate_tn(float tstart, float tstop, size_t ntpoints, float* tn)
{
	size_t i;
	float dt = (tstop - tstart)/ntpoints;
	for (i = 0; i < ntpoints; ++i)
	{
		tn[i] = tstart + dt*i;
	}
}



int main(int argc, char** argv)
{
	float fce = 2*PI*atof(argv[1]);
	float fm  = 2*PI*atof(argv[2]);
	float a0  = atof(argv[3]);
	size_t nh  = atoi(argv[4]);
	float tstart = atof(argv[5]);
	float tstop = atof(argv[6]);
	size_t ntpoints = atoi(argv[7]);
	printf("Start the programm!\n");
	printf("Parameters:\n");
	printf("fce = %f Hz,\nfm = %f Hz,\na0 = %f V/cm,\nnh = %d,\ntstart = %f sec.,\ntstop = %f sec.,\nntpoints = %d\n", fce/2/PI, fm/2/PI, a0, (int)nh, tstart, tstop, (int)ntpoints);
	
  
	float* an = (float*)malloc(sizeof(float)*nh);
	float* fn = (float*)malloc(sizeof(float)*nh);
	float* vnp = (float*)malloc(sizeof(float)*nh);
	float* vnm = (float*)malloc(sizeof(float)*nh);
	float* enm = (float*)malloc(sizeof(float)*nh);
	float* enp = (float*)malloc(sizeof(float)*nh);
	float* tn =  (float*)malloc(sizeof(float)*ntpoints);
	float* vt = (float*)malloc(sizeof(float)*ntpoints);
	float* et = (float*)malloc(sizeof(float)*ntpoints);
		
	float* d_vnm = NULL;
	float* d_vnp = NULL;
	float* d_enm = NULL;
	float* d_enp = NULL;
	float* d_tn = NULL;
	float* d_fn = NULL;
	float* d_fce = NULL;
	float* d_sinp = NULL;
	float* d_sinm = NULL;
	float* d_vt = NULL;
	float* d_et = NULL;
	
	size_t i = 0;
	FILE* to;
	
	cudaMalloc(&d_fce, sizeof(float));
	cudaMalloc(&d_vnm, sizeof(float)*nh);
	cudaMalloc(&d_vnp, sizeof(float)*nh);
	cudaMalloc(&d_sinp, sizeof(float)*nh);
	cudaMalloc(&d_sinm, sizeof(float)*nh);
	cudaMalloc(&d_fn, sizeof(float)*nh);
	cudaMalloc(&d_tn, sizeof(float)*ntpoints);
	cudaMalloc(&d_vt, sizeof(float)*ntpoints);
	cudaMalloc(&d_et, sizeof(float)*ntpoints);
	cudaMalloc(&d_enm, sizeof(float)*nh);
	cudaMalloc(&d_enp, sizeof(float)*nh);

	generate_fn(fm, nh, fn);
	generate_an(a0, nh, an);
	generate_vn(an, fn, fce, nh, vnp, vnm);
	generate_en(an, fn, fce, nh, enp, enm);
	generate_tn(tstart, tstop, ntpoints, tn);

	cudaMemcpy(d_vnm, vnm, sizeof(float)*nh, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vnp, vnp, sizeof(float)*nh, cudaMemcpyHostToDevice);
	cudaMemcpy(d_enm, enm, sizeof(float)*nh, cudaMemcpyHostToDevice);
	cudaMemcpy(d_enp, enp, sizeof(float)*nh, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fn, fn, sizeof(float)*nh, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fce, &fce, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tn, tn, sizeof(float)*ntpoints, cudaMemcpyHostToDevice);

	global_calculate_vt<<<ntpoints/MAX_THREADS, MAX_THREADS>>>(d_fn, d_fce, d_vnp, d_vnm, nh, d_tn, d_vt);
	global_calculate_et<<<ntpoints/MAX_THREADS, MAX_THREADS>>>(d_fn, d_fce, d_enp, d_enm, nh, d_tn, d_et);
	
	cudaMemcpy(vt, d_vt, sizeof(float)*ntpoints, cudaMemcpyDeviceToHost);
	cudaMemcpy(et, d_et, sizeof(float)*ntpoints, cudaMemcpyDeviceToHost);
	
	to = fopen("vt.dat", "w");
	for(i = 0; i < ntpoints; ++i)
	{
		fprintf(to, "%e\t%e\n", tn[i], vt[i]);
	}
	fclose(to);

	to = fopen("et.dat", "w");
	for(i = 0; i < ntpoints; ++i)
	{
		fprintf(to, "%e\t%e\n", tn[i], et[i] + sin(fce*tn[i]));
	}
	fclose(to);
	

	free(vnp);
	free(vnm);
	free(an);
	free(fn);
	free(tn);
	free(vt);
	free(et);
	free(enm);
	free(enp);
	cudaFree(d_vnm);
	cudaFree(d_vnp);
	cudaFree(d_enm);
	cudaFree(d_enp);
	cudaFree(d_sinm);
	cudaFree(d_sinp);
	cudaFree(d_fn);
	cudaFree(d_tn);
	cudaFree(d_fce);
	cudaFree(d_vt);

	printf("The programm is done!\n");
}
