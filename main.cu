#include "stdlib.h"
#include "cudpp.h"
#include "math.h"
#include "stdio.h"
#include "libconfig.h"
#include "curand.h"
#include "curand_kernel.h"
#include "linux/limits.h"
#include "limits.h"
#include "cufft.h"

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
	float pulse_ratio;
}pulse_t;

typedef struct simulation_str
{
	float tstart;
	float tstop;
	unsigned int nh;
	float dt;
	unsigned int nelectrons;
	unsigned int ncells;
	float z1;
	float z2;
	float plasma_density;
	int static_ez;
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
	unsigned int max_gpu_threads;
	unsigned int rseed;
	char* vsave;
	unsigned int save_every_n;
}global_setting_t;


__device__ float diff(float k, float dx)
{
    float a = k * dx;
    return sin(a)/a;
}

__device__ float diff2(float k, float dx)
{
    float a = diff(k, dx);
    return a*a;
}

void set_speed_maxwell_cuda(curandGenerator_t cuda_r, float* d_v, float sigma, unsigned int nelectrons)
{
	CURAND_CALL(curandGenerateNormal(cuda_r, d_v, nelectrons, 0, sigma));
}

__global__ void transform_uniform_distribution(float* d_p, float min,  float max)
{
    unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
    d_p[n] = (max - min)*d_p[n] + min;
}

void set_pos_uniform_cuda(curandGenerator_t cuda_r, float* d_r, float z1, float z2, unsigned int nelectrons, unsigned int max_threads)
{
	CURAND_CALL(curandGenerateUniform(cuda_r, d_r, nelectrons));
	transform_uniform_distribution<<<nelectrons/max_threads, max_threads>>>(d_r, z1, z2);
}

void calculate_v2(float* vx, float* vy, unsigned int ntpoints, float* v)
{
	unsigned int i;
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
	setting = config_lookup(config, "pulse.pulse_ratio");
	pulse->pulse_ratio = (float)config_setting_get_float(setting);
	return 0;
}

int get_global_config(config_t* config, global_setting_t* global_settings)
{
	config_setting_t* setting = config_lookup(config, "global.filename");
	global_settings->save_file = (char*)config_setting_get_string(setting);	
	setting = config_lookup(config, "global.max_gpu_threads");
	global_settings->max_gpu_threads = (unsigned int)config_setting_get_int64(setting);
	setting = config_lookup(config, "global.savedir");
	global_settings->savedir = (char*)config_setting_get_string(setting);
	setting = config_lookup(config, "global.rseed");
	global_settings->rseed = (unsigned int)config_setting_get_int64(setting);
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
	simulation->nh = (unsigned int)config_setting_get_int64(setting);
	setting = config_lookup(config, "simulation.nelectrons");
	simulation->nelectrons = (unsigned int)config_setting_get_int64(setting);
	setting = config_lookup(config, "simulation.ncells");
	simulation->ncells = (unsigned int)config_setting_get_int64(setting);
	setting = config_lookup(config, "simulation.z1");
	simulation->z1 = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.z2");
	simulation->z2 = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.plasma_density");
	simulation->plasma_density = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.static_ez");
	simulation->static_ez = config_setting_get_bool(setting);
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



__global__ void kernel_generate_ambmfm(float am, float fm, float pulse_ratio, float nh, float* d_fm, float* d_am, float* d_bm)
{
	float tm = 2*PI/fm;
	float tm1 = tm/pulse_ratio;
	unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i==0)
	{
		d_bm[i] = am*(2*tm1-tm)/tm;
		d_am[i] = 0;
	}
	else
	{
		d_am[i] = 2*am*(1-cos(i*fm*tm1))/(i*PI);
		d_bm[i] = 2*am*sin(i*fm*tm1)/(i*PI);
	}
	d_fm[i] = fm*i;	
}

__global__ void generate_ameandr(float* d_am, float* d_bm, float* d_fm, float t,  unsigned int nh, float* d_a)
{
	unsigned int i;
	float a = 0;
	for (i = 0; i < nh; ++i)
	{
		a += d_am[i]*sin(d_fm[i]*t) + d_bm[i]*cos(d_fm[i]*t);
	}
	*d_a = a; 
}


int my_read_config_file(char* file, config_t* config)
{
	config_init(config);
	return config_read_file(config, file);
}


__global__ void global_associate_electrons_with_cells(float* d_z, float cellsize, unsigned int* d_association)
{
	unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int ncell = floor(d_z[n]/cellsize);
	d_association[n] = ncell;
}

__device__ float get_ex_field(float A0, float pulse_duration, 
			      float z, unsigned int nt, 
			      float z0, float dz,
			      float* am, float* tn)
{
	float az = (fabs(z - z0) <= dz) ? 1 : 0;  
	float at = (tn[nt] - pulse_duration) < 0 ? 1 : 0;
	return 0.5*az*at*(am[nt] + A0);
	
}

__global__ void update_ex_field(float A0, float* d_am, float fce, float dt, 
				unsigned int nt, float pulse_duration, float z0, float pulse_dz, 
				float cellsize, float* d_ex, float q, float m)
{
	unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
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
	unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
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
					    unsigned int* d_associate, 
					    float fce, float delta)
{
	unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
	float vxt = d_vx[n];
	float vyt = d_vy[n];
	float dt = delta;
	float tfce = fce;
	unsigned int ncell = d_associate[n];
	rotate_particle(&vxt, &vyt,  d_fx[ncell], tfce, dt);
	d_vx[n] = vxt;
	d_vy[n] = vyt;
	d_vz[n] += d_fz[ncell]*delta;
	d_z[n] += d_vz[n]*dt; 
	
}

__global__ void postproc(float* d_z, float* d_vx, float* d_vy, 
			 float zmin, float zmax, float sigma,  curandState* d_states)
{
	unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int nr = threadIdx.x;
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
					   unsigned int* d_associate, 
					   float* d_m, unsigned int* d_n,
					   unsigned int nelectrons, float mass)
{
	unsigned int ncell = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int i;
	unsigned int ne = 0;
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

void dump(char* savedir, char* filename, unsigned int n, float* d_vx, float* d_vy, 
	  unsigned int*  d_cell_electron_association, float* d_m, unsigned int* d_n, 
	  unsigned int nelectrons, unsigned int ncells, float m, float cellsize, float z1, float dt, 
	  unsigned int max_threads)
{
	FILE* to;
	float* momentum = (float*)malloc(sizeof(float)*ncells);
	unsigned int i;
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
	unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
	curand_init(rseed, n, 0, &states[n]);
}

__global__  void do_calculate_rho_cuda(unsigned int* d_associate, 
				       unsigned int* d_rho)
{	
	unsigned int nelectron = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int ncell = d_associate[nelectron];
	atomicInc(&d_rho[ncell], UINT_MAX);
}

__global__ void generate_kn(float k0, float dx, float* d_k)
{
	unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
	float k = k0*(n+1);
	d_k[n] = k*k*diff2(k,dx/2);
}

__global__ void copy_rho_to_cufft(unsigned int* d_rho, cufftComplex* data, float coeff)
{
	unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
	data[n] = make_cuFloatComplex (coeff*(float)d_rho[n],0);
}

__global__ void copy_cufft_to_phi(float* d_phi, cufftComplex* data)
{
	unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
	cufftComplex thisd = data[n];
	d_phi[n] = cuCrealf(thisd);
}

__global__ void poisson_harmonics_transform(float* d_k, cufftComplex* data, unsigned int nharm)
{
	unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
	cufftComplex thisd = data[n+1];
	float r;
	float i;
	r = 4*PI*cuCrealf(thisd)/d_k[n];
	i = 4*PI*cuCimagf(thisd)/d_k[n];
	data[n+1] =  make_cuFloatComplex (r,i);
	data[nharm/2 + n] = cuConjf(data[nharm/2 - n]);
}

__global__ void zero_harm_hack_for_fftdata(cufftComplex* data)
{
	data[0] =  make_cuFloatComplex(0,0);
}

__global__ void calculate_ez_cuda(float* d_phi, float* d_ez, unsigned int ncells, float q, float m, float cellsize)
{
	
	unsigned int n = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int nl = n - 1;
	unsigned int nr = n + 1;
	float qm = q/m;
	if (n==0)
	{
		nl = ncells-1;
		nr = 1;
	}
	if (n==ncells-1)
	{
		nl = ncells-2;
		nr = 0;
	}
	d_ez[n]= -qm*(d_phi[nr] - d_phi[nl])/2/cellsize/ncells;
}

void update_ez_cuda(cufftHandle plan, unsigned int* d_rho, float* d_k, float* d_ez, float* d_phi, 
		    cufftComplex* data, unsigned int ncells, unsigned int max_threads, 
		    float dens_coeff, float q, float m, float cellsize)
{
	copy_rho_to_cufft<<<ncells/max_threads, max_threads>>>(d_rho, data, dens_coeff);
	cufftExecC2C(plan, data, data, CUFFT_FORWARD);
	poisson_harmonics_transform<<<ncells/max_threads/2, max_threads>>>(d_k, data, ncells);
	zero_harm_hack_for_fftdata<<<1,1>>>(data);
	cufftExecC2C(plan, data, data, CUFFT_INVERSE);
	copy_cufft_to_phi<<<ncells/max_threads, max_threads>>>(d_phi, data);
	calculate_ez_cuda<<<ncells/max_threads, max_threads>>>(d_phi, d_ez, ncells, q,m, cellsize);
}

void dump_vperp(char* savedir, char* savefile, unsigned int n, 
		float* d_vx, float* d_vy, float* vx, float* vy, 
		unsigned int* d_associate, unsigned int* associate,
		unsigned int nelectrons, unsigned int ncells, float dz)
{
	unsigned int i;
	char fullpath[PATH_MAX];
	CUDA_CALL(cudaMemcpy(vx, d_vx, sizeof(float)*nelectrons, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(vy, d_vy, sizeof(float)*nelectrons, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(d_associate, associate, sizeof(float)*nelectrons, cudaMemcpyDeviceToHost));
	sprintf(fullpath, "%s/%s_%d.dat", savedir, savefile, n);
	FILE* to;
	for (i = 0; i < ncells; ++i)
	{
		sprintf(fullpath, "%s/%s_%d_%d_vx.dat", savedir, savefile, n, i);
		to = fopen(fullpath, "w");
		fwrite(vx, sizeof(float), nelectrons, to);
		fclose(to);
		sprintf(fullpath, "%s/%s_%d_%d_vy.dat", savedir, savefile, n, i);
		to = fopen(fullpath, "w");
		fwrite(vy, sizeof(float), nelectrons, to);
		fclose(to);
		sprintf(fullpath, "%s/%s_%d_%d_association.dat", savedir, savefile, n, i);
		to = fopen(fullpath, "w");
		fwrite(associate, sizeof(long int), nelectrons, to);
		fclose(to);
	}
	printf("Particle speed and distribution dumped: %d\n", i);
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
	unsigned int nh  = simulation.nh;
	float tstart = simulation.tstart;
	float tstop = simulation.tstop;
	float dt = simulation.dt;
	unsigned int ntpoints = ceil((tstop-tstart)/dt);
	float pulse_duration = pulse.T;
	float pulse_ratio = pulse.pulse_ratio;
	unsigned int ncells = simulation.ncells;
	unsigned int nelectrons = simulation.nelectrons;
	unsigned int max_threads = global_settings.max_gpu_threads;
	unsigned int rseed = global_settings.rseed;
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
	int static_ez = simulation.static_ez;

	printf("Allocate memory\n");
	
	curandState* d_rstates;
	CUDA_CALL(cudaMalloc(&d_rstates, sizeof(curandState)*max_threads));
	setup_rstates<<<1,max_threads>>>(d_rstates, rseed);
	CURAND_CALL(curandCreateGenerator(&cuda_r, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(cuda_r, rseed));

	float* d_vx = NULL;
	float* d_vy = NULL;
	float* d_vz = NULL;
	float* d_z = NULL;
	float* d_ex = NULL;
	unsigned int* d_cell_electron_association = NULL;
	unsigned int* d_buffer = NULL;

	unsigned int* n = (unsigned int*)malloc(sizeof(unsigned int)*ncells);
	float* d_am = NULL;
	float* d_bm;
	float* d_fm = NULL;
	float* d_a = NULL;
	float* d_m = NULL;
	unsigned int* d_n = NULL;
	unsigned int* d_rho;
	float* ez =  (float*)malloc(sizeof(float)*ncells);
	float* d_ez;
	float* d_phi;
	float density_simulation_coeff = q*plasma_density*(z2 - z1)/nelectrons/dz;
	float* d_k;
	cufftComplex *d_fft_data;
	cufftHandle plan;

	CUDA_CALL(cudaMalloc((void**)&d_fft_data, sizeof(cufftComplex)*ncells));
	cufftPlan1d(&plan, ncells, CUFFT_C2C, 1);

	CUDA_CALL(cudaMalloc(&d_vx, sizeof(float)*nelectrons));
	CUDA_CALL(cudaMalloc(&d_vy, sizeof(float)*nelectrons));
	CUDA_CALL(cudaMalloc(&d_vz, sizeof(float)*nelectrons));
	CUDA_CALL(cudaMalloc(&d_z, sizeof(float)*nelectrons));
	CUDA_CALL(cudaMalloc(&d_ex, sizeof(float)*ncells));
	CUDA_CALL(cudaMalloc(&d_am, sizeof(float)*nh));
	CUDA_CALL(cudaMalloc(&d_bm, sizeof(float)*nh));
	CUDA_CALL(cudaMalloc(&d_fm, sizeof(float)*nh));
	CUDA_CALL(cudaMalloc(&d_a, sizeof(float)));
	CUDA_CALL(cudaMalloc(&d_m, sizeof(float)*ncells));
	CUDA_CALL(cudaMalloc(&d_n, sizeof(unsigned int)*ncells));
	CUDA_CALL(cudaMalloc(&d_rho, sizeof(unsigned int)*ncells));
	CUDA_CALL(cudaMalloc(&d_ez, sizeof(float)*ncells));
	CUDA_CALL(cudaMalloc(&d_phi, sizeof(float)*ncells));
	CUDA_CALL(cudaMalloc(&d_k, sizeof(float)*ncells/2));
	CUDA_CALL(cudaMalloc(&d_cell_electron_association, sizeof(unsigned int)*nelectrons));
	CUDA_CALL(cudaMalloc(&d_buffer, sizeof(unsigned int)*nelectrons));

	printf("Preparing the initial data\n");

	CUDA_CALL(cudaMemset(d_n, 0, sizeof(float)*ncells));
	CUDA_CALL(cudaMemset(d_m, 0, sizeof(float)*ncells));
	CUDA_CALL(cudaMemset(d_ez, 0, sizeof(float)*ncells));
	
	kernel_generate_ambmfm<<<1,nh>>>(Am, fm, pulse_ratio, nh, d_fm, d_am, d_bm);

	set_speed_maxwell_cuda(cuda_r, d_vx, sigma, nelectrons);
	set_speed_maxwell_cuda(cuda_r, d_vy, sigma, nelectrons);
	set_speed_maxwell_cuda(cuda_r, d_vz, sigma, nelectrons);
	set_pos_uniform_cuda(cuda_r, d_z, z1, z2, nelectrons, max_threads);
	generate_kn<<<ncells/max_threads/2, max_threads>>>(2*PI/(z2-z1), dz, d_k);
	
	unsigned int i = 0;	
		
	printf("Start the calculations!\n");
	
	for(i = 0; i < ntpoints; ++i)
	{
	
		//printf("cycle number: %i\n", i);
		generate_ameandr<<<1, 1>>>(d_am, d_bm, d_fm, i*dt,  nh, d_a);
		update_ex_field<<<ncells/max_threads, max_threads>>>(A0, d_a, fce, dt, i, 
								     pulse_duration, 
								     pulse_z0, pulse_dz, dz, 
								     d_ex, q, m);
	
		postproc<<<nelectrons/max_threads, max_threads>>>(d_z, d_vx, d_vy, z1, z2, sigma, d_rstates);
		global_associate_electrons_with_cells<<<nelectrons/max_threads, max_threads>>>(d_z,
											       dz, 
											       d_cell_electron_association);
		if(static_ez)
		{
			CUDA_CALL(cudaMemset(d_rho, 0, sizeof(unsigned int)*ncells));
			do_calculate_rho_cuda<<<nelectrons/max_threads, max_threads>>>(d_cell_electron_association, d_rho);
			update_ez_cuda(plan, d_rho, d_k, d_ez, d_phi, d_fft_data, ncells, 
				       max_threads, density_simulation_coeff,q,m,dz );
		}
		
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
	CUDA_CALL(cudaFree(d_bm));
	CUDA_CALL(cudaFree(d_fm));
	CUDA_CALL(cudaFree(d_a));
	CUDA_CALL(cudaFree(d_cell_electron_association));
	CUDA_CALL(cudaFree(d_m));
	CUDA_CALL(cudaFree(d_n));
	CUDA_CALL(cudaFree(d_rho));
	CUDA_CALL(cudaFree(d_ez));
	CUDA_CALL(cudaFree(d_rstates));
	CUDA_CALL(cudaFree(d_fft_data));
	CUDA_CALL(cudaFree(d_phi));
	CUDA_CALL(cudaFree(d_buffer));
	free(n);
	free(ez);
	cufftDestroy(plan);
	CURAND_CALL(curandDestroyGenerator(cuda_r));
	config_destroy (&configuration);
	printf("The programm is done!\n");
}
