#include "poisson1d.h"
#include "gsl/gsl_errno.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"

#define PI (3.14159265)

double diff(double k, double dx)
{
    double a = k * dx;
    return sin(a)/a;
}

double diff2(double k, double dx)
{
    double a = diff(k, dx);
    return a*a;
}

int forward_fft(gsl_fft_complex_wavetable* wavetable,
                gsl_fft_complex_workspace* workspace,
                double* data, double* fftdata,
                int size)
{
    int i = 0;
    for (i = 0; i < 2*size; i++)
    {
        fftdata[i] = data[i];
    }
    return gsl_fft_complex_forward(fftdata, 1, size,
                                  wavetable, workspace);

}

int reverse_fft(gsl_fft_complex_wavetable* wavetable,
                gsl_fft_complex_workspace* workspace,
                double* data, double* fftdata,
                int size)
{
    int i = 0;
    for (i = 0; i < 2*size; i++)
    {
        data[i] = fftdata[i];
    }
    return gsl_fft_complex_inverse(data, 1, size,
                                       wavetable, workspace);
}

int poisson1d_init_fft(int size, float dx, poisson1d* cfft)
{
    int i = 0;
    int j = 1;
	#ifdef DEBUG
    printf("DEBUG: init fft\n");
	#endif
    cfft->dx = (double)dx;
    cfft->k0 = 2*PI/((size-1)*dx); //size-1!! 
    cfft->size = size;
    cfft->rho = NULL;
    cfft->fftrho = NULL;
    cfft->phi = NULL;
    cfft->fftphi = NULL;
    cfft->wavetable = NULL;
    cfft->workspace = NULL;
    cfft->kappa = NULL;
    cfft->K = NULL;

    cfft->K = (double*) malloc(sizeof (double) * size);
    cfft->k = (double*) malloc(sizeof (double) * size);
    cfft->kappa = (double*) malloc(sizeof (double) * size);
    cfft->rho = (double*) malloc(sizeof (double) * 2*size);
    cfft->fftrho = (double*) malloc(sizeof (double) * 2*size);
    cfft->phi = (double*) malloc(sizeof (double) * 2*size);
    cfft->fftphi = (double*) malloc(sizeof (double) * 2*size);
    cfft->wavetable = gsl_fft_complex_wavetable_alloc(size);
    cfft->workspace = gsl_fft_complex_workspace_alloc(size);


    if ((cfft->rho == NULL)
        || (cfft->fftrho == NULL)
        || (cfft->phi == NULL)
        || (cfft->fftphi == NULL)
        || (cfft->wavetable == NULL)
        || (cfft->workspace == NULL)
        || (cfft->K == NULL)
        || (cfft->kappa == NULL))
    {
        return EXIT_FAILURE;
    }
    for (i = 0; i < size; i++)
    {

        REAL(cfft->rho,i) = 0;
        IMAGE(cfft->rho,i) = 0;
        REAL(cfft->fftrho,i) = 0;
        IMAGE(cfft->fftrho,i) = 0;
        REAL(cfft->phi,i) = 0;
        IMAGE(cfft->phi,i) = 0;
        REAL(cfft->fftphi,i) = 0;
        IMAGE(cfft->fftphi,i) = 0;

    }
    for (i = 1; i <= size/2; i++)
    {
         double k = (cfft->k0)*i;
         cfft->k[i] = k;
         cfft->kappa[i] = k*diff(k, dx);
         cfft->K[i] = k*k*diff2(k, dx/2);
         cfft->K[size-i] = - cfft->K[i];
         cfft->k[size - i] = -cfft->k[i];
    }

    return EXIT_SUCCESS;
}

int poisson1d_free_fft(poisson1d* cfft)
{
#ifdef DEBUG
    printf("DEBUG: free fft\n");
#endif
    free(cfft->rho);
    free(cfft->fftrho);
    free(cfft->phi);
    free(cfft->fftphi);
    free(cfft->kappa);
    free(cfft->K);
    free(cfft->k);
    gsl_fft_complex_wavetable_free(cfft->wavetable);
    gsl_fft_complex_workspace_free(cfft->workspace);
    return EXIT_SUCCESS;
}

int poisson1d_make_fftrho_from_real_rho(poisson1d* p1d)
{
    return forward_fft(p1d->wavetable, p1d->workspace, p1d->rho, p1d->fftrho, p1d->size);
}

int poisson1d_make_fftphi_from_fftrho(poisson1d* p1d)
{
    int i = 0;
    REAL(p1d->fftphi,0) = 0;
    IMAGE(p1d->fftphi,0) = 0;
    for (i = 1; i <= p1d->size/2; i++)
    {
        REAL(p1d->fftphi, i) = 4*PI*(REAL(p1d->fftrho,i))/p1d->K[i];
        IMAGE(p1d->fftphi, i) = 4*PI*(IMAGE(p1d->fftrho,i))/p1d->K[i];
    }
    for(i = p1d->size/2; i < p1d->size; i++)
    {
        REAL(p1d->fftphi, i) =  REAL(p1d->fftphi, p1d->size - i);
        IMAGE(p1d->fftphi, i) = - IMAGE(p1d->fftphi,p1d->size - i);
    }
    return EXIT_SUCCESS;
}

int poisson1d_make_phi_from_fftphi(poisson1d* p1d)
{
    return reverse_fft(p1d->wavetable, p1d->workspace, p1d->phi, p1d->fftphi, p1d->size);
}

int poisson1d_load_rho(poisson1d* p1d, float* from_rho)
{
    int i = 0;
    for (i = 0; i < p1d->size; i++)
    {
        REAL(p1d->rho, i) = (double)from_rho[i];
        IMAGE(p1d->rho,i) = 0.0; //It kinda says us: "What are you starring at?!!"
    }
	return i;
}

 int poisson1d_unload_phi(poisson1d* p1d, float* to_phi)
 {
     int i = 0;
    for (i = 0; i < p1d->size; i++)
    {
        to_phi[i] = (float)REAL(p1d->phi,i);
    }
	return i;
 }

 int poisson1d_calc_phi_from_rho(poisson1d* p1d)
 {
     poisson1d_make_fftrho_from_real_rho(p1d);
     poisson1d_make_fftphi_from_fftrho(p1d);
     poisson1d_make_phi_from_fftphi(p1d);
	 return EXIT_SUCCESS;
 }


 int poisson1d_cut_fftrho_harmonics(poisson1d* p1d, int cut)
 {
     int i = 0;
     for (i = cut; i < p1d->size; i++)
     {
         p1d->fftrho[i] = 0;
     }
	 return i;
 }
 
int poisson1d_cutoff_fftrho_exponentially(poisson1d* p1d, float k_lim)
{
    int i = 0;
    double efactor;
    #pragma omp parallel for
    for (i = 1; i <= p1d->size/2; ++i)
    {
        efactor = exp(-(p1d->k[i]/k_lim)*(p1d->k[i]/k_lim));
        REAL(p1d->fftrho, i) = REAL(p1d->fftrho, i)*efactor;
        IMAGE(p1d->fftrho, i) = IMAGE(p1d->fftrho, i)*efactor;
    }
    return EXIT_SUCCESS;
}
#ifdef DEBUG

int do_print_data_into_file(FILE* to, double* data, int size)
{
    int i = 0;
#ifdef DEBUG
    printf("DEBUG: dump poisson data into file\n");
#endif
    for (i = 0; i < size; i++)
    {
        fprintf(to, "%e\n", data[i]);
    }
    return EXIT_SUCCESS;
}

int poisson1d_print_data_into_file(char* file, double* data, int size)
{
    FILE* to = fopen(file, "w");
    int result = do_print_data_into_file(to, data, size);
    fclose(to);
    return result;
}
#endif
