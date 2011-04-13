#include "stdlib.h"
#include "math.h"
#include "stdio.h"
#include "cudpp.h"
#include "libconfig.h"
#include "curand.h"
#include "curand_kernel.h"
#include "linux/limits.h"
#include "omp.h"
#include "poisson1d.h"

#define PI (3.14159265)
#define EV_IN_ERGS (1.60217646e-12)

#define CURAND_CALL(x)  { if((x) != CURAND_STATUS_SUCCESS) {		\
			printf("Error %d at %s:%d\n", x, __FILE__,__LINE__); \
			exit(EXIT_FAILURE);}}

#define CUDA_CALL(x) { if((x) != cudaSuccess) { \
			printf("Cuda error %d at %s:%d: ", x, __FILE__,__LINE__); \
			printf("%s\n", cudaGetErrorString(x));		\
			exit(EXIT_FAILURE);}}


typedef struct pulse_cfg_str
{
	float A0;
	float Am;
	float T;
	float fm;
	float fce;
	float z0;
	float dz;
}pulse_t;

typedef struct simulation_str
{
	float tstart;
	float tstop;
	long long nh;
	float dt;
	long long nelectrons;
	long long ncells;
	float z1;
	float z2;
	float plasma_density;
}simulation_t;

typedef struct electron_str
{
	float T;
	float m;
	float q;
}
electron_t;


typedef struct global_settings_str
{
	char* save_file;
	char* savedir;
	long long max_gpu_threads;
	long long rseed;
}global_setting_t;

void set_speed_maxwell_cuda(curandGenerator_t cuda_r, float* d_v, float sigma, long long nelectrons)
{
	CURAND_CALL(curandGenerateNormal(cuda_r, d_v, nelectrons, 0, sigma));
}

__global__ void transform_uniform_distribution(float* d_p, float min,  float max)
{
    long long n = blockDim.x*blockIdx.x + threadIdx.x;
    d_p[n] = (max - min)*d_p[n] + min;
}

void set_pos_uniform_cuda(curandGenerator_t cuda_r, float* d_r, float z1, float z2, long long nelectrons, long long max_threads)
{
	CURAND_CALL(curandGenerateUniform(cuda_r, d_r, nelectrons));
	transform_uniform_distribution<<<nelectrons/max_threads, max_threads>>>(d_r, z1, z2);
}

void calculate_v2(float* vx, float* vy, long long ntpoints, float* v)
{
	long long i;
	for (i = 0; i < ntpoints; ++i)
	{
		v[i] = vx[i]*vx[i] + vy[i]*vy[i];
	}
}

int get_pulse_config(config_t* config, pulse_t* pulse)
{
	config_setting_t* setting = config_lookup(config, "pulse.A0");
	pulse->A0 = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "pulse.Am");
	pulse->Am = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "pulse.fce");
	pulse->fce = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "pulse.fm");
	pulse->fm = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "pulse.T");
	pulse->T = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "pulse.z0");
	pulse->z0 = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "pulse.dz");
	pulse->dz = (float)config_setting_get_float(setting);
	return 0;
}

int get_global_config(config_t* config, global_setting_t* global_settings)
{
	config_setting_t* setting = config_lookup(config, "global.filename");
	global_settings->save_file = (char*)config_setting_get_string(setting);	
	setting = config_lookup(config, "global.max_gpu_threads");
	global_settings->max_gpu_threads = config_setting_get_int64(setting);
	setting = config_lookup(config, "global.savedir");
	global_settings->savedir = (char*)config_setting_get_string(setting);
	setting = config_lookup(config, "global.rseed");
	global_settings->rseed = config_setting_get_int64(setting);
	return 0;
}

int get_simulation_config(config_t* config, simulation_t* simulation)
{
	config_setting_t* setting = config_lookup(config, "simulation.tstart");
	simulation->tstart = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.tstop");
	simulation->tstop = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.dt");
	simulation->dt = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.nharm");
	simulation->nh = config_setting_get_int64(setting);
	setting = config_lookup(config, "simulation.nelectrons");
	simulation->nelectrons = config_setting_get_int64(setting);
	setting = config_lookup(config, "simulation.ncells");
	simulation->ncells = config_setting_get_int64(setting);
	setting = config_lookup(config, "simulation.z1");
	simulation->z1 = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.z2");
	simulation->z2 = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.plasma_density");
	simulation->plasma_density = (float)config_setting_get_float(setting);
	return 0;
}

int get_electron_config(config_t* config, electron_t* electron)
{
	config_setting_t* setting = config_lookup(config, "electron.T");
	electron->T = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "electron.m");
	electron->m = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "electron.q");
	electron->q = (float)config_setting_get_float(setting);
	return  0;
}

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


__global__ void global_calculate_et(float* fn, float* fce, float* enp, float* enm, long long nh, float* tn, float* et)
{      
	long long nt = blockDim.x*blockIdx.x + threadIdx.x;
	float t = tn[nt];
	long long i;
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

__global__ void kernel_generate_amfm(float am, float fm, float nh, float* d_fm, float* d_am)
{
	long long i = blockDim.x*blockIdx.x + threadIdx.x;
	d_am[i] = 4*am/PI/(2*i + 1);
	d_fm[i] = fm*(2*i + 1);	
}

__global__ void generate_ameandr(float* d_am, float* d_fm, float t,  long long nh, float* d_a)
{
	long long i;
	float a = 0;
	for (i = 0; i < nh; ++i)
	{
		a += d_am[i]*sin(d_fm[i]*t);
	}
	*d_a = a; 
}

void generate_en(float* an, float* fn, float fce,
		 long long nh, float* enp, float* enm)
{
	long long i;
	for (i = 0; i < nh; ++i)
	{     
		enm[i] = 0.5*an[i];
		enp[i] = -0.5*an[i];
	}	
}


int my_read_config_file(char* file, config_t* config)
{
	config_init(config);
	return config_read_file(config, file);
}


__global__ void global_associate_electrons_with_cells(float* d_z, float cellsize, long long* d_association)
{
	long long n = blockDim.x*blockIdx.x + threadIdx.x;
	long long ncell = floor(d_z[n]/cellsize);
	d_association[n] = ncell;
}

__device__ float get_ex_field(float A0, float pulse_duration, 
			      float z, long long nt, 
			      float z0, float dz,
			      float* am, float* tn)
{
	float az = (fabs(z - z0) <= dz) ? 1 : 0;  
	float at = (tn[nt] - pulse_duration) < 0 ? 1 : 0;
	return 0.5*az*at*(am[nt] + A0);
	
}

__global__ void update_ex_field(float A0, float* d_am, float fce, float dt, 
				long long nt, float pulse_duration, float z0, float pulse_dz, 
				float cellsize, float* d_ex, float q, float m)
{
	long long n = blockDim.x*blockIdx.x + threadIdx.x;
        float z = (n + 0.5)*cellsize;
	float t = nt*dt;
	float sfce = sin(fce*t);
	float az = (fabs(z - z0) <= pulse_dz) ? 1 : 0; 
	float at = (t - pulse_duration) < 0 ? 1 : 0;
	//d_ex[n] = get_ex_field(A0, pulse_duration, z, nt, z0, dz, d_am, tn)*sfce*q/m;
	d_ex[n] = at*az*0.5*(A0 + *d_am)*sfce*q/m;
}

__global__ void kernel_generate_tn(float tstart, float dt, float* d_tn)
{
	long long n = blockDim.x*blockIdx.x + threadIdx.x;
	d_tn[n] = tstart + n*dt;
}


__device__ void rotate_particle(float *vx, float* vy,  float E, float fce, float delta)
{
	float vx_m;
	float vy_m;
	
	float S = sin(fce*delta);
	float C = cos(fce*delta);
	
	vx_m = *vx + delta*E/2.;
	vy_m = *vy;
	
	*vx = (vx_m*C + vy_m*S) + delta*E/2.;
	*vy = (-vx_m*S + vy_m*C);
}

__global__ void trace_electrons_single_step(float* d_vx, float* d_vy, float* d_vz,
					    float* d_z, float* d_fx, float* d_fz, 
					    long long* d_associate, 
					    float fce, float delta)
{
	long long n = blockDim.x*blockIdx.x + threadIdx.x;
	float vxt = d_vx[n];
	float vyt = d_vy[n];
	float dt = delta;
	float tfce = fce;
	long long ncell = d_associate[n];
	rotate_particle(&vxt, &vyt,  d_fx[ncell], tfce, dt);
	d_vx[n] = vxt;
	d_vy[n] = vyt;
	d_vz[n] += d_fz[ncell]*delta;
	d_z[n] += d_vz[n]*dt; 
	
}

__global__ void postproc(float* d_z, float* d_vx, float* d_vy, 
			 float zmin, float zmax, float sigma,  curandState* d_states)
{
	long long n = blockDim.x*blockIdx.x + threadIdx.x;
	long long nr = threadIdx.x;
	float L = zmax - zmin;
	curandState rstate = d_states[nr];
	if(d_z[n] > zmax)
	{
		d_z[n] -= L;
		d_vx[n] = sigma*curand_normal(&rstate);
		d_vy[n] = sigma*curand_normal(&rstate);
	}
	if(d_z[n] < zmin)
	{
		d_z[n] += L;
		d_vx[n] = sigma*curand_normal(&rstate);
		d_vy[n] = sigma*curand_normal(&rstate);
	
	}
	d_states[nr]=rstate;
}

__global__ void calculate_momentum_zdistribution(float* d_vx, float* d_vy, 
					   long long* d_associate, 
					   float* d_m, long long* d_n,
					   long long nelectrons, float mass)
{
	long long ncell = blockDim.x*blockIdx.x + threadIdx.x;
	long long i;
	long long ne = 0;
	float m = 0;
	for (i = 0; i < nelectrons; ++i)
	{
		if (d_associate[i] == ncell)
		{
			float vx = d_vx[i];
			float vy = d_vy[i];
			ne++;
			m +=  vx*vx + vy*vy;
		}
	}
	d_n[ncell] = ne;
	d_m[ncell] = 0.5*mass*m/ne;
}

void dump(char* savedir, char* filename, long long n, float* d_vx, float* d_vy, 
	  long long*  d_cell_electron_association, float* d_m, long long* d_n, 
	  long long nelectrons, long long ncells, float m, float cellsize, float z1, float dt, 
	  long long max_threads)
{
	FILE* to;
	float* momentum = (float*)malloc(sizeof(float)*ncells);
	long long i;
	char filenamei[PATH_MAX];
	sprintf(filenamei, "%s/%s_%d.dat", savedir, filename, n);
	to = fopen(filenamei, "w");
	calculate_momentum_zdistribution<<<ncells/max_threads, max_threads>>>(d_vx, d_vy, d_cell_electron_association, d_m, d_n, nelectrons, m);
	cudaMemcpy(momentum, d_m, sizeof(float)*ncells, cudaMemcpyDeviceToHost);
	
	for (i = 0; i < ncells; ++i)
	{
		fprintf(to, "%e\t%e\n", z1 + (i+0.5)*cellsize,  momentum[i]);
	}
	free(momentum);
	fclose(to);
	printf("file %d dumped\n", n);
}

__global__ void setup_rstates(curandState* states, unsigned long rseed)
{
	size_t n = blockDim.x*blockIdx.x + threadIdx.x;
	curand_init(rseed, n, 0, &states[n]);
}

__global__ void calculate_rho(long long* d_associate, long long* d_rho, long long nelectrons)
{
	long long ncell = blockDim.x*blockIdx.x + threadIdx.x;
	long long i;
	long long rho = 0;
	for (i = 0; i < nelectrons; ++i)
	{
		if (d_associate[i]==ncell)
		{
			rho++;
		}
	}
	d_rho[ncell] = rho;
}

void update_ez(poisson1d* poisson, long long* d_associate, long long* d_rho, 
	       long long* rho, float* d_ez, float* ez, float* phi, 
	       long long ncells, long long nelectrons, long long max_threads, float dcoeff,
	       float cellsize, float q, float m)
{
	calculate_rho<<<ncells/max_threads, max_threads>>>(d_associate, d_rho, nelectrons);
	CUDA_CALL(cudaMemcpy(rho, d_rho, sizeof(long long)*ncells, cudaMemcpyDeviceToHost));
	float qm = q/m;
	long long i;
#pragma omp parallel for
	for (i = 0; i < ncells; ++i)
	{
		ez[i] = (float)dcoeff*rho[i];
	}
	poisson1d_load_rho(poisson, ez);
	poisson1d_calc_phi_from_rho(poisson);
	poisson1d_unload_phi(poisson, phi);
#pragma omp parallel for
	for (i = 1; i < ncells - 1; ++i) 
	{
		ez[i]= -qm*(phi[i+1] - phi[i-1])/2/cellsize;
	}
	ez[0] = -qm*(phi[1]-phi[ncells-1])/2/cellsize;
	ez[ncells-1] = -qm*(phi[0]-phi[ncells-2])/2/cellsize;
	CUDA_CALL(cudaMemcpy(d_ez, ez, sizeof(float)*ncells, cudaMemcpyHostToDevice));
}

int main(int argc, char** argv)
{
	char* config_file = argv[1];
	printf("Read the configuration in %s\n", config_file);
	config_t configuration;
	pulse_t pulse;
	simulation_t simulation;
	global_setting_t global_settings;
	electron_t electron;
	my_read_config_file(config_file, &configuration);
	
	get_pulse_config(&configuration, &pulse);
	get_simulation_config(&configuration, &simulation);
	get_global_config(&configuration, &global_settings);
	get_electron_config(&configuration, &electron);
	
	float fce = 2*PI*pulse.fce;
	float fm  = 2*PI*pulse.fm;
	float Am  = pulse.Am;
	float A0 = pulse.A0;
	float pulse_dz = pulse.dz;
	float pulse_z0 = pulse.z0;
	long long nh  = simulation.nh;
	float tstart = simulation.tstart;
	float tstop = simulation.tstop;
	float dt = simulation.dt;
	long long ntpoints = ceil((tstop-tstart)/dt);
	float pulse_duration = pulse.T;
	long long ncells = simulation.ncells;
	long long nelectrons = simulation.nelectrons;
	long long max_threads = global_settings.max_gpu_threads;
	long long rseed = global_settings.rseed;
	float z1 = simulation.z1;
	float z2 = simulation.z2;
	float dz = (z2-z1)/ncells;
	float q = electron.q;
	float m = electron.m;
	float Te = electron.T;
	float sigma = sqrt(EV_IN_ERGS*Te/m);
	float plasma_density = simulation.plasma_density;
	curandGenerator_t cuda_r;
	char* filename = global_settings.save_file;
	char* savedir = global_settings.savedir;
	poisson1d poisson;

	printf("Allocate memory\n");
	
	curandState* d_rstates;
	CUDA_CALL(cudaMalloc(&d_rstates, sizeof(curandState)*max_threads));
	setup_rstates<<<1,max_threads>>>(d_rstates, rseed);
	CURAND_CALL(curandCreateGenerator(&cuda_r, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(cuda_r, rseed));

	poisson1d_init_fft(ncells, dz, &poisson);
	float* d_vx = NULL;
	float* d_vy = NULL;
	float* d_vz = NULL;
	float* d_z = NULL;
	float* d_ex = NULL;
	long long* d_cell_electron_association = NULL;

	long long* n = (long long*)malloc(sizeof(long long)*ncells);
	float* d_am = NULL;
	float* d_fm = NULL;
	float* d_a = NULL;
	float* d_m = NULL;
	long long* d_n = NULL;
	long long* d_rho;
	long long* rho = (long long*)malloc(sizeof(long long)*ncells);
	float* ez =  (float*)malloc(sizeof(float)*ncells);
	float* phi =  (float*)malloc(sizeof(float)*ncells);
	float* d_ez;
	float density_simulation_coeff = q*plasma_density*(z2 - z1)/nelectrons/dz;

	CUDA_CALL(cudaMalloc(&d_vx, sizeof(float)*nelectrons));
	CUDA_CALL(cudaMalloc(&d_vy, sizeof(float)*nelectrons));
	CUDA_CALL(cudaMalloc(&d_vz, sizeof(float)*nelectrons));
	CUDA_CALL(cudaMalloc(&d_z, sizeof(float)*nelectrons));
	CUDA_CALL(cudaMalloc(&d_ex, sizeof(float)*ncells));
	CUDA_CALL(cudaMalloc(&d_am, sizeof(float)*nh));
	CUDA_CALL(cudaMalloc(&d_fm, sizeof(float)*nh));
	CUDA_CALL(cudaMalloc(&d_a, sizeof(float)));
	CUDA_CALL(cudaMalloc(&d_m, sizeof(float)*ncells));
	CUDA_CALL(cudaMalloc(&d_n, sizeof(long long)*ncells));
	CUDA_CALL(cudaMalloc(&d_rho, sizeof(long long)*ncells));
	CUDA_CALL(cudaMalloc(&d_ez, sizeof(float)*ncells));
	CUDA_CALL(cudaMalloc(&d_cell_electron_association, sizeof(long long)*nelectrons));

	printf("Preparing the initial data\n");

	CUDA_CALL(cudaMemset(d_n, 0, sizeof(float)*ncells));
	CUDA_CALL(cudaMemset(d_m, 0, sizeof(float)*ncells));
	
	kernel_generate_amfm<<<1,nh>>>(Am, fm, nh, d_fm, d_am);

	set_speed_maxwell_cuda(cuda_r, d_vx, sigma, nelectrons);
	set_speed_maxwell_cuda(cuda_r, d_vy, sigma, nelectrons);
	set_speed_maxwell_cuda(cuda_r, d_vz, sigma, nelectrons);
	set_pos_uniform_cuda(cuda_r, d_z, z1, z2, nelectrons, max_threads);
	long long i = 0;	
		
	printf("Start the calculations!\n");
		

	for(i = 0; i < ntpoints; ++i)
	{
		//printf("cycle number: %i\n", i);
		generate_ameandr<<<1, 1>>>(d_am, d_fm, i*dt,  nh, d_a);
		update_ex_field<<<ncells/max_threads, max_threads>>>(A0, d_a, fce, dt, i, 
								     pulse_duration, 
								     pulse_z0, pulse_dz, dz, 
								     d_ex, q, m);
	
		postproc<<<nelectrons/max_threads, max_threads>>>(d_z, d_vx, d_vy, z1, z2, sigma, d_rstates);
		global_associate_electrons_with_cells<<<nelectrons/max_threads, max_threads>>>(d_z,
											       dz, 
											       d_cell_electron_association);
		update_ez(&poisson, d_cell_electron_association, d_rho, 
			  rho, d_ez, ez, phi, ncells, nelectrons, max_threads, 
			  density_simulation_coeff,
			  dz, q, m);
		trace_electrons_single_step<<<nelectrons/max_threads, max_threads>>>(d_vx, d_vy, d_vz, 
										     d_z, d_ex, d_ez, 
										     d_cell_electron_association, 
										     fce, dt);
		if ((i%100) == 0)
		{
			dump(savedir, filename, i/100, d_vx, d_vy, d_cell_electron_association, d_m,
			     d_n, nelectrons, ncells, m, dz, z1, dt, max_threads);
		}
	}



	printf("The calculations are done!\n");
	printf("Free the memory\n");

	CUDA_CALL(cudaFree(d_vx));
	CUDA_CALL(cudaFree(d_vy));
	CUDA_CALL(cudaFree(d_vz));
	CUDA_CALL(cudaFree(d_z));
	CUDA_CALL(cudaFree(d_ex));
	CUDA_CALL(cudaFree(d_am));
	CUDA_CALL(cudaFree(d_fm));
	CUDA_CALL(cudaFree(d_a));
	CUDA_CALL(cudaFree(d_cell_electron_association));
	CUDA_CALL(cudaFree(d_m));
	CUDA_CALL(cudaFree(d_n));
	CUDA_CALL(cudaFree(d_rho));
	CUDA_CALL(cudaFree(d_ez));
	CUDA_CALL(cudaFree(d_rstates));
	free(n);
	free(rho);
	free(ez);
	free(phi);
	CURAND_CALL(curandDestroyGenerator(cuda_r));
	config_destroy (&configuration);
	poisson1d_free_fft(&poisson);
	printf("The programm is done!\n");
}
