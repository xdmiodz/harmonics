#include "stdio.h"
#include "stdlib.h"
#include "linux/limits.h"
#include "string.h"
#include "math.h"
#include "omp.h" 

void usage()
{
	printf("df: an utility to construct distribution function using dump files.\n");
	printf("usage: df savedir savefile  dumpdir dumpfile startcell stopcell nvsteps nelectrons ntpoints\n");
	printf("savedir - the dir where result shall be saved\n");
	printf("savefile - the file name template for the results\n");
	printf("dumpdir - the dir where dump files are saved\n");
	printf("dumpfile - the file name template of the dummp files\n");
	printf("startcell, stopcell - the bounds of the region where the df shall be calculated\n");
	printf("nvsteps - the total number of speed steps for calculation of the df\n");
	printf("nelectrons - the total number of particles saved in a dump file\n");
	printf("ntpoints - the total number of timeponts saved in the dump files\n");
}

unsigned int select_particles(float* from, float* to, unsigned int* ass, 
			      unsigned int nstart, unsigned int nstop, 
			      unsigned int nelectrons)
{
	unsigned int i;
	unsigned int k = 0;
	#pragma omp parallel for
	for(i = 0; i < nelectrons; ++i)
	{
		if((ass[i] >= nstart) && (ass[i] <= nstop))
		{
			#pragma omp critical
			{
				to[k]=from[i];
				k++;
			}
		}
	}
	return k;
}

int comp(const void* m1, const void* m2)
{
	if (*(float*)m1 > *(float*)m2)
		return 1;
	if (*(float*)m1 < *(float*)m2)
		return -1;
	return 0;
	
}

void gdf(float* vsel, unsigned int* f, float* vf, unsigned int nv, unsigned int nvsteps)
{
	qsort(vsel, nv, sizeof(float), comp);
	float vmin = vsel[0];
	float vmax = vsel[nv-1];
	float dv = (vmax - vmin)/nvsteps;
	unsigned int i;
	unsigned int k;
	memset(f, 0, sizeof(unsigned int)*nvsteps);
	float vt = 0;

	for (i = 0; i < nvsteps; ++i)
	{
		vt = vmin + i*dv;
		vf[i] = vt + 0.5*dv;
		for (k = 0; k < nv; ++k)
		{
			if(fabs(vsel[k]-vt) < dv/2)
			{
				f[i]++;
			}
		}				
	}	
}

int main(int argc, char** argv)
{
	if (0 == strcmp(argv[1], "help"))
	{
		usage();
		return 0;
	}
	char* savedir = argv[1];
	char* savefile = argv[2];
	char* dumpdir = argv[3];
	char* dumpfile = argv[4];
	unsigned int nstart = atoi(argv[5]);
	unsigned int nstop = atoi(argv[6]);
	unsigned int n = nstart - nstop + 1;
	unsigned int nvsteps = atoi(argv[7]);
	unsigned int nelectrons = atoi(argv[8]);
	unsigned int ntpoints = atoi(argv[9]);
	unsigned int i;
	FILE* from;
	FILE* to;
	char fullsavename[PATH_MAX];
	char fulldumpname[PATH_MAX];
	float* v = (float*)malloc(sizeof(float)*nelectrons);
	float* vf = (float*)malloc(sizeof(float)*nvsteps);
	unsigned int* f = (unsigned int*)malloc(sizeof(unsigned int)*nvsteps);
	unsigned int nvf;
	float* vsel = (float*)malloc(sizeof(float)*nelectrons);
	unsigned int* ass = (unsigned int*)malloc(sizeof(unsigned int)*nelectrons);
	unsigned int j;
	for (i = 0; i < ntpoints; ++i)
	{
		sprintf(fulldumpname, "%s/%s_%d_vperp.dat", dumpdir, dumpfile, i);
		from = fopen(fulldumpname, "r");
		fread(v, sizeof(float), nelectrons, from);
		fclose(from);
		sprintf(fulldumpname, "%s/%s_%d_association.dat", dumpdir, dumpfile, i);
		from = fopen(fulldumpname, "r");
		fread(ass, sizeof(unsigned int), nelectrons, from);
		fclose(from);
		nvf = select_particles(v, vsel, ass, nstart, nstop, nelectrons);
		gdf(vsel, f, vf, nvf, nvsteps);
		sprintf(fulldumpname, "%s/%s_%d.dat", savedir, savefile, i);
		to = fopen(fulldumpname, "w");
		for(j = 0; j < nvsteps; ++j)
		{
			fprintf(to, "%d\t%e\n", f[j], vf[j]);
		}
		fclose(to);
		
	}
	free(ass);
	free(v);
	free(vsel);
	free(vf);
	free(f);
}
