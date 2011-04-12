#include "stdlib.h"
#include "math.h"
#include "stdio.h"
#include "cudpp.h"
#include "libconfig.h"
#include "curand.h"

#define PI (3.14159265)
#define EV_IN_ERGS (1.60217646e-12)

#define CURAND_CALL(x)  { if((x) != CURAND_STATUS_SUCCESS) {		\
			printf("Error %d at %s:%d\n", x, __FILE__,__LINE__); \
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
	float Tstart;
	float Tstop;
	float sample_dt;
	long long nh;
	long long ntpoints;
	long long nelectrons;
	long long ncells;
	float z1;
	float z2;
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
	char* v_save_file;
	long long max_gpu_threads;
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
	global_settings->v_save_file = (char*)config_setting_get_string(setting);	
	setting = config_lookup(config, "global.max_gpu_threads");
	global_settings->max_gpu_threads = config_setting_get_int64(setting);
	return 0;
}

int get_simulation_config(config_t* config, simulation_t* simulation)
{
	config_setting_t* setting = config_lookup(config, "simulation.Tstart");
	simulation->Tstart = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.Tstop");
	simulation->Tstop = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.dT");
	simulation->sample_dt = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.nharm");
	simulation->nh = config_setting_get_int64(setting);
	setting = config_lookup(config, "simulation.ntpoints");
	simulation->ntpoints = config_setting_get_int64(setting);
	setting = config_lookup(config, "simulation.nelectrons");
	simulation->nelectrons = config_setting_get_int64(setting);
	setting = config_lookup(config, "simulation.ncells");
	simulation->ncells = config_setting_get_int64(setting);
	setting = config_lookup(config, "simulation.z1");
	simulation->ncells = (float)config_setting_get_float(setting);
	setting = config_lookup(config, "simulation.z2");
	simulation->ncells = (float)config_setting_get_float(setting);
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

__global__ void global_calculate_vt(float* fn, float* fce, float* vnp, float* vnm, long long nh, float* tn, float* vt)
{      
	long long nt = blockDim.x*blockIdx.x + threadIdx.x;
	float t = tn[nt];
	long long i;
	float kfce = *fce;
	float sinp;
	float sinm;
	
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

__global__ void global_calculate_increase_vt(float* fce, float* tn, float* pulse_an, float* vt)
{
	long long nt = blockDim.x*blockIdx.x + threadIdx.x;
	float t = tn[nt];
	float kfce = *fce;
	vt[nt] = pulse_an[nt]*t*sin(kfce*t);
}

__global__ void global_calculate_v3(float C, float *fce, float* tn, float* v3)
{
	size_t nt = blockDim.x*blockIdx.x + threadIdx.x;
	float t = tn[nt];
	float kfce = *fce;
	v3[nt] = -(C/kfce)*sin(kfce*t);
}


__global__ void global_calc_the_constant(float* fce, float* fn, float* vnp, float* vnm, float* cn)
{
	size_t nh = blockDim.x*blockIdx.x + threadIdx.x;
	float fhn = fn[nh];
	float tfce = *fce;
	float vnpn = vnp[nh];
	float vnmn = vnm[nh];
	cn[nh] = (fhn + tfce)*vnpn + (fhn - tfce)*vnmn;
} 

__global__ void global_calculate_vt2(float* vt, float* vt2)
{
	long long nt = blockDim.x*blockIdx.x + threadIdx.x;
	float vtn = vt[nt];
	vt2[nt] = vtn*vtn;
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

__global__ void global_get_all_speeds_alltogether(float *d_vi, float* d_vn, float* d_v)
{
	long long nt = blockDim.x*blockIdx.x + threadIdx.x;
	d_v[nt] = (d_vn[nt] + d_vi[nt]);
}

void generate_am(float Am, float fm, long long nh, float* am, float* tn, long long ntpoints)
{
	long long i;
	long long j;
	float* fn = (float*)malloc(sizeof(float)*nh);
	float* an = (float*)malloc(sizeof(float)*nh);
	float a = Am*4/PI;
	for (i = 0; i < nh; ++i)
	{
		an[i] = a/(2*i+1);
		fn[i] = fm*(2*i+1);
	}
	for(i = 0; i < ntpoints; ++i)
	{
		am[i] = 0;
		for (j = 0; j < nh; ++j)
		{
			am[i] += an[j]*sin(fn[j]*tn[i]);
		}
	}
}

void generate_fn(float fm, long long nh, float* fn)
{
	long long i;
	for (i = 0; i < nh; ++i)
	{
		fn[i] = fm*(2*i+1);
	}
}

void generate_vn(float* an, float* fn, float fce,
		 long long nh, float* vnp, float* vnm)
{
	long long i;
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
		 long long nh, float* enp, float* enm)
{
	long long i;
	for (i = 0; i < nh; ++i)
	{     
		enm[i] = 0.5*an[i];
		enp[i] = -0.5*an[i];
	}	
}

void generate_tn(float tstart, float tstop, long long ntpoints, float* tn)
{
	long long i;
	float dt = (tstop - tstart)/ntpoints;
	for (i = 0; i < ntpoints; ++i)
	{
		tn[i] = tstart + dt*i;
	}
}

void generate_pulse_amp(float pulse_duration, long long ntpoints, float* tn, 
			float A0, float* am, float fce, float* pulse_an)
{
	long long i;
	for(i = 0; i < ntpoints; ++i)
	{
		if(pulse_duration >= tn[i])
		{
			pulse_an[i] =  0.5*(A0 + am[i])*sin(fce*tn[i]);
		}
		else
		{
			pulse_an[i] =  am[i]*sin(fce*tn[i]);
		}
	}
}

int my_read_config_file(char* file, config_t* config)
{
	config_init(config);
	return config_read_file(config, file);
}

__global__ void global_generate_cells_bounds(float z1, float dz, 
					     float* d_lcell, float* d_rcell)
{
	long long n = blockDim.x*blockIdx.x + threadIdx.x;
	d_lcell[n] = z1 + n*dz;
	d_rcell[n] = z1 + (n + 1)*dz;
}

__global__ void global_associate_electrons_with_cells(float* d_z, float* d_lcell, 
						      float* d_rcell, long long ncells,
						      long long* d_association)
{
	long long n = blockDim.x*blockIdx.x + threadIdx.x;
	float z = d_z[n];
	long long i;
	for (i = 0; i < ncells; ++i)
	{
		if((z >= d_lcell[i]) && (z <= d_rcell[i]))
		{
			d_association[n] = i;
		}
	}
}

__device__ float get_ex_field(float A0, float pulse_duration, 
			      float z, float t, 
			      float z0, float dz,
			      float* am, float* fm, 
			      float nh, float fce)
{
	long long i;
	float a = (fabs(z - z0) <= dz) ? 1 : 0;
	float ameandr = 0;
	for(i = 0; i < nh; i++)
	{
		ameandr = am[i]*sin(fm[i]*t);
	}
	return 0.5*a*(ameandr + A0)*sin(fce*t);
	
}

__global__ void update_ex_field(float A0, long long nharm, 
				float* d_am, float* d_fm, float fce, float t, 
				float pulse_duration, float z0, float dz, 
				float* d_lcell, float* d_rcell, float* d_ex)
{
	long long n = blockDim.x*blockIdx.x + threadIdx.x;
        float z = 0.5*(d_lcell[n] + d_rcell[n]);
	d_ex[n] = get_ex_field(A0, pulse_duration, z, t, z0, dz, d_am, d_fm, nharm, fce);
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
					    float* d_z, float* d_ex, long long* d_associate, 
					    float fce, float delta)
{
	long long n = blockDim.x*blockIdx.x + threadIdx.x;
	float vxt = d_vx[n];
	float vyt = d_vy[n];
	float dt = delta;
	float tfce = fce;
	long long ncell = d_associate[n];
	rotate_particle(&vxt, &vyt,  d_ex[ncell], tfce, dt);
	d_z[n] += d_vz[n]*dt; 
}

__global__ void calculate_v2_zdistribution(float* d_vx, float* d_vy, 
					   long long* d_associate, float* d_v2)
{
	long long n = blockDim.x*blockIdx.x + threadIdx.x;
	long long ncell = d_associate[n];
	float vx = d_vx[ncell];
	float vy = d_vy[ncell];
	d_v2[n] += vx*vx + vy*vy;
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
	long long nh  = simulation.nh;
	float tstart = simulation.Tstart;
	float tstop = simulation.Tstop;
	long long ntpoints = simulation.ntpoints;
	float pulse_duration = pulse.T;
	long long ncells = simulation.ncells;
	long long nelectrons = simulation.nelectrons;
	curandGenerator_t cuda_r;
	CURAND_CALL(curandCreateGenerator(&cuda_r, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(cuda_r, 1234ULL));

	long long* cell_electron_association = (long long*)malloc(sizeof(long long)*nelectrons);
	
	float sample_dt = simulation.sample_dt;
	float* d_vx = NULL;
	float* d_vy = NULL;
	float* d_vz = NULL;
	float* d_ex = NULL;
	float* d_cell_electron_association = NULL;
	float* d_lcell = NULL;
	float* d_rcell = NULL;
	float* v = (float*)malloc(sizeof(float)*nelectrons);
	float* tn = (float*)malloc(sizeof(float)*ntpoints);

	cudaMalloc(&d_vx, sizeof(float)*nelectrons);
	cudaMalloc(&d_vy, sizeof(float)*nelectrons);
	cudaMalloc(&d_vz, sizeof(float)*nelectrons);
	cudaMalloc(&d_ex, sizeof(float)*ncells);
	cudaMalloc(&d_lcell, sizeof(float)*ncells);
	cudaMalloc(&d_rcell, sizeof(float)*ncells);

	cudaMalloc(&d_cell_electron_association, sizeof(long long)*nelectrons);
	
	float* am = (float*)malloc(sizeof(float)*ntpoints);
	float* pulse_an = (float*)malloc(sizeof(float)*ncells);
	
	long long i = 0;
	FILE* to;	
		
	printf("Start the calculations!\n");	
     
	
	printf("The calculations are done!\n");

	printf("Save the data!\n");

	cudaFree(d_vx);
	cudaFree(d_vy);
	cudaFree(d_vz);
	cudaFree(d_ex);
	cudaFree(d_lcell);
	cudaFree(d_rcell);

	config_destroy (&configuration);
	printf("The programm is done!\n");
}
