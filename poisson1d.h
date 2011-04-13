/*
 * File:   poisson1d.h
 * Author: oddi
 *
 * Created on October 22, 2010, 5:13 PM
 */


#ifndef _POISSON1D_H
#define	_POISSON1D_H

#include "gsl/gsl_fft_complex.h"

#define REAL(data,n)  ((data)[2*(n)])
#define IMAGE(data,n) ((data)[2*(n)+1])


#ifdef DEBUG
#define POISSON1D_REALMODE 1
#define POISSON1D_IMAGEMODE 2
#define POISSON1D_MIXMODE 3
#endif


typedef struct struct_poisson1d {
    int size;
    double dx;
    double k0;
    double* k;
    double* kappa;
    double* K;
    double* rho;
    double* fftrho;
    double* phi;
    double* fftphi;
    gsl_fft_complex_wavetable* wavetable;
    gsl_fft_complex_workspace* workspace;
}poisson1d;

int poisson1d_init_fft(int size, float dx, poisson1d* cfft);
int poisson1d_make_fftrho_from_real_rho(poisson1d*);
int poisson1d_make_fftphi_from_fftrho(poisson1d*);
int poisson1d_make_phi_from_fftphi(poisson1d*);
int poisson1d_free_fft(poisson1d* cfft);

int poisson1d_cut_fftrho_harmonics(poisson1d* p1d, int cut);
int poisson1d_cutoff_fftrho_exponentially(poisson1d* p1d, float k_lim);
int poisson1d_load_rho(poisson1d* p1d, float* from_rho);
int poisson1d_calc_phi_from_rho(poisson1d* p1d);
int poisson1d_unload_phi(poisson1d* p1d, float* to_phi);

#ifdef DEBUG
int poisson1d_print_data_into_file(char* file, double* data, int size);
#endif


#endif	/* _POISSON1D_H */
