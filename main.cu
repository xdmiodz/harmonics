#include "stdlib.h"
#include "math.h"
#include "stdio.h"
#include "cudpp.h"
#include "libconfig.h"
#include "linux/limits.h"

#define PI (3.14159265)


typedef struct pulse_cfg_str
{
	float A0;
	float Am;
	float T;
	float fm;
	float fce;
	
}pulse_t;

typedef struct simulation_str
{
	float Tstart;
	float Tstop;
	float sample_dt;
	long long nh;
	long long ntpoints;
}simulation_t;

typedef struct global_settings_str
{
	char* v_save_file;
	long long max_gpu_threads;
}global_setting_t;

void rotate_particle(float *vx, float* vy,  float E, float fce, float delta)
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

void advance_particle(float delta, float* vx, float* vy, float fce, 
		      float* En, long long ntpoints)
{
	long long i;
	float vx_temp;
	float vy_temp;
	for(i = 1; i < ntpoints; ++i)
	{
		vx_temp = vx[i-1];
		vy_temp = vy[i-1]; 
		rotate_particle(&vx_temp, &vy_temp, En[i], fce, delta);
		vx[i]=vx_temp;
		vy[i]=vy_temp;
	}
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
	return 0;
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

int main(int argc, char** argv)
{
	char* config_file = argv[1];
	printf("Read the configuration in %s\n", config_file);
	config_t configuration;
	pulse_t pulse;
	simulation_t simulation;
	global_setting_t global_settings;
	my_read_config_file(config_file, &configuration);
	
	get_pulse_config(&configuration, &pulse);
	get_simulation_config(&configuration, &simulation);
	get_global_config(&configuration, &global_settings);
	
	float fce = 2*PI*pulse.fce;
	float fm  = 2*PI*pulse.fm;
	float Am  = pulse.Am;
	float A0 = pulse.A0;
	long long nh  = simulation.nh;
	float tstart = simulation.Tstart;
	float tstop = simulation.Tstop;
	long long ntpoints = simulation.ntpoints;
	float pulse_duration = pulse.T;

	
	float sample_dt = simulation.sample_dt;
	float* vx = (float*)malloc(sizeof(float)*ntpoints);
	float* vy = (float*)malloc(sizeof(float)*ntpoints);
	float* v = (float*)malloc(sizeof(float)*ntpoints);
	float* tn = (float*)malloc(sizeof(float)*ntpoints);

	float* am = (float*)malloc(sizeof(float)*ntpoints);
	float* pulse_an = (float*)malloc(sizeof(float)*ntpoints);
	
	long long i = 0;
	FILE* to;	
	
	memset((void*)vx, 0, sizeof(float)*ntpoints);
	memset((void*)vy, 1, sizeof(float)*ntpoints);
	
	printf("Start the calculations!\n");	
      

	generate_tn(tstart, tstop, ntpoints, tn);
	generate_am(Am, fm, nh, am,  tn, ntpoints);
	generate_pulse_amp(pulse_duration, ntpoints, tn,  A0, am, fce, pulse_an);
	advance_particle(tn[1]-tn[0], vx, vy, fce, pulse_an, ntpoints);
	calculate_v2(vx,vy, ntpoints, v);
	
	printf("The calculations are done!\n");

	printf("Save the data!\n");

	to = fopen("pulse.dat", "w");
	for(i = 0; i < ntpoints; ++i)
	{
		fprintf(to, "%e\t%e\n", tn[i], pulse_an[i]);
	}
	fclose(to);

	to = fopen("v.dat", "w");
	for(i = 0; i < ntpoints; ++i)
	{
		fprintf(to, "%e\t%e\n", tn[i], v[i]);
	}
	fclose(to);
	
	to = fopen("vx.dat", "w");
	for(i = 0; i < ntpoints; ++i)
	{
		fprintf(to, "%e\t%e\n", tn[i], vx[i]);
	}
	fclose(to);

	to = fopen("vy.dat", "w");
	for(i = 0; i < ntpoints; ++i)
	{
		fprintf(to, "%e\t%e\n", tn[i], vy[i]);
	}
	fclose(to);

	printf("The saving is done!\n");
	free(vx);
	free(vy);
	free(v);
	free(tn);
	free(am);
	free(pulse_an);
	config_destroy (&configuration);
	printf("The programm is done!\n");
}
