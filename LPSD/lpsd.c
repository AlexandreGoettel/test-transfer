/********************************************************************************
    lpsd.c
			  
    2003, 2004 by Michael Troebs, mt@lzh.de and Gerhard Heinzel, ghh@mpq.mpg.de

    calculate spectra from time series using discrete Fourier 
    transforms at frequencies equally spaced on a logarithmic axis
    
    lpsd does everything except user interface and data output
    
 ********************************************************************************/
#define SINCOS

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <fftw3.h>
#include "hdf5.h"

#include "config.h"
#include "ask.h"
#include "IO.h"
#include "genwin.h"
#include "debug.h"
#include "lpsd.h"
#include "misc.h"
#include "errors.h"

/*
20.03.2004: http://www.caddr.com/macho/archives/iolanguage/2003-9/549.html
*/
#ifndef __linux__
#include <windows.h>
struct timezone
{
  int tz_minuteswest;
  int tz_dsttime;
};
void
gettimeofday (struct timeval *tv, struct timezone *tz
	      __attribute__ ((unused)))
{
  long int count = GetTickCount ();
  tv->tv_sec = (int) (count / 1000);
  tv->tv_usec = (count % 1000) * 1000;
}

#else
#include <sys/time.h>		/* gettimeofday, timeval */
#endif

#ifdef __linux__
extern double round (double x);
/*
gcc (GCC) 3.3.1 (SuSE Linux) gives warning: implicit declaration of function `round'
without this line - math.h defines function round, but its declaration seems to be
missing in math.h 
*/

#else
/*
int round (double x) {
  return ((int) (floor (x + 0.5)));
}
*/
#endif

/********************************************************************************
 * 	global variables						   	
 ********************************************************************************/
static int nread;
static double winsum;
static double winsum2;
static double nenbw;		/* normalized equivalent noise bandwidth */
static double *dwin;		/* pointer to window function for FFT */

/********************************************************************************
 * 	functions								
 ********************************************************************************/


// Get mean over N first values of int sequence
double
get_mean (int* values, int N) {
    double _sum = 0;
    register int i;
    for (i = 0; i < N; i++) {
        _sum += values[i];
    }
    return _sum / N;
}


// @brief recursive function to count set bits
int
countSetBits (int n)
{
    // base case
    if (n == 0)
        return 0;
    else
        // if last bit set add 1 else add 0
        return (n & 1) + countSetBits(n >> 1);
}


int
get_next_power_of_two (int n)
{
    int output = n;
    if (!(countSetBits(n) == 1 || n == 0))
      output = (int) pow(2, (int) log2(n) + 1);
    return output;
}


double*
stride_over_array (double *data, int N, int stride, int offset, double *output)
{
    for (int i = offset; i < N; i += stride) *(output++) = data[i];
}


// Get the segment length as a function of the frequency bin j
// Rounded to nearest integer.
// TODO: replace with call to nffts?
int
get_N_j(int j, double fsamp, double fmin, double fmax, int Jdes) {
    double g = log(fmax) - log(fmin);  // TODO: could consider making g part of cfg
    return round (fsamp/fmin * exp(-j*g / (Jdes - 1.)) / (exp(g / (Jdes - 1.)) - 1.));
}

// Get the frequency of bin j
// TODO: replace with call to fspec?
double
get_f_j(int j, double fmin, double fmax, int Jdes) {
    double g = log(fmax) - log(fmin);  // TODO: could consider making g part of cfg
    return fmin*exp(j*g / (Jdes - 1.));
}


static void
getDFT2 (int nfft, double bin, double fsamp, double ovlp, double *rslt,
         int *avg, struct hdf5_contents *contents)
{
  /* Configure variables for DFT */
  int max_samples_in_memory = 5*6577770;  // Around 500 MB //TODO: this shouldn't be hard-coded!
//  int max_samples_in_memory = 512;  // tmp
  if (max_samples_in_memory > nfft) max_samples_in_memory = nfft; // Don't allocate more than you need

  /* Allocate data and window memory segments */
  double *strain_data_segment = (double*) xmalloc(max_samples_in_memory * sizeof(double));
  double *window = (double*) xmalloc(2*max_samples_in_memory * sizeof(double));
  assert(window != 0 && strain_data_segment != 0);

  //////////////////////////////////////////////////
  /* Calculate DFT over separate memory windows */
  int window_offset, count;
  int memory_unit_index = 0;
  int remaining_samples = nfft;
  int nsum = floor(1+(nread - nfft) / floor(nfft * (1.0 - (double) (ovlp / 100.))));
  int tmp = (nsum-1)*floor(nfft * (1.0 - (double) (ovlp / 100.)))+nfft;
  if (tmp == nread) nsum--;  /* Adjust for edge case */

  double dft_results[2*nsum];  /* Real and imaginary parts of DFTs */
  memset(dft_results, 0, 2*nsum*sizeof(double));

  while (remaining_samples > 0)
  {
    if (remaining_samples > max_samples_in_memory)
    {
      count = max_samples_in_memory;
      remaining_samples -= max_samples_in_memory;
    } else {
      count = remaining_samples;
      remaining_samples = 0;
    }
    window_offset = memory_unit_index * max_samples_in_memory;
    memory_unit_index++;

    // Calculate window
    makewinsincos_indexed(nfft, bin, window, &winsum, &winsum2, &nenbw,
                          window_offset, count, window_offset == 0);

    // Loop over data segments
    int start = 0;
    register int _nsum = 0;
    hsize_t data_count[1] = {count};
    while (start + nfft < nread)
    {
      // Load data
      hsize_t data_offset[1] = {start + window_offset};
      read_from_dataset(contents, data_offset, data_count, strain_data_segment);

      // Calculate DFT
      register int i;
      for (i = 0; i < count; i++)
      {
        dft_results[_nsum*2] += window[i*2] * strain_data_segment[i];
        dft_results[_nsum*2 + 1] += window[i*2 + 1] * strain_data_segment[i];
      }
      start += nfft * (1.0 - (double) (ovlp / 100.));  /* go to next segment */
      _nsum++;
    }
  }

  /* Sum over dft_results to get total */
  register int i;
  double total = 0;  /* Running sum of DFTs */
  for (i = 0; i < nsum; i++)
  {
    total += dft_results[i*2]*dft_results[i*2] + dft_results[i*2+1]*dft_results[i*2+1];
  }
  //////////////////////////////////////////////////

  /* Return result */
  rslt[0] = total / nsum;
  
  /* This sets the variance to zero. This is not true, but we are not using the variance. */
  rslt[1] = 0;
  
  rslt[2] = rslt[0];
  rslt[3] = rslt[1];
  rslt[0] *= 2. / (fsamp * winsum2);	/* power spectral density */
  rslt[1] *= 2. / (fsamp * winsum2);	/* variance of power spectral density */
  rslt[2] *= 2. / (winsum * winsum);	/* power spectrum */
  rslt[3] *= 2. / (winsum * winsum);	/* variance of power spectrum */

  *avg = nsum;

  /* clean up */
  xfree(window);
  xfree(strain_data_segment);
}


/*
	calculates paramaters for DFTs
	output
		fspec		frequencies in spectrum
		bins		bins for DFTs
		nffts		dimensions for DFTs
 ********************************************************************************
 	Naming convention	source code	publication
        i		    j
        fres	    r''
        ndft	    L(j)
        bin		    m(j)
 */
static void
calc_params (tCFG * cfg, tDATA * data)
{
  double fres, f, bin, g;
  int i, i0, ndft;

  g = log ((*cfg).fmax / (*cfg).fmin);
  i = (*cfg).nspec * (*cfg).iter;
  i0 = i;
  f = (*cfg).fmin * exp (i * g / ((*cfg).Jdes - 1.));
  while (f <= (*cfg).fmax && i / (*cfg).nspec < (*cfg).iter + 1)
   {
      fres = f * (exp (g / ((*cfg).Jdes - 1.)) - 1);
      ndft = round ((*cfg).fsamp / fres);
      bin = (f / fres);
      (*data).fspec[i - i0] = f;
      (*data).nffts[i - i0] = ndft;
      (*data).bins[i - i0] = bin;
      i++;
      f = (*cfg).fmin * exp (i * g / ((*cfg).Jdes - 1.));
  }
  (*cfg).nspec = i - i0;
  (*cfg).fmin = (*data).fspec[0];
  (*cfg).fmax = (*data).fspec[(*cfg).nspec - 1];
}

void
calculate_lpsd (tCFG * cfg, tDATA * data)
{
  int k;			/* 0..nspec */
  int k_start = 0;		/* N. lines in save file. Post fail start point */
  char ch;			/* For scanning through checkpointing file */
  int Nsave = (*cfg).nspec / 100; /* Frequency of data checkpointing */
  int j; 			/* Iteration variables for checkpointing data */
  FILE * file1;			/* Output file, temp for checkpointing */
  double rslt[4];		/* rslt[0]=PSD, rslt[1]=variance(PSD) rslt[2]=PS rslt[3]=variance(PS) */
  double progress;

  struct timeval tv;
  double start, now, print;

  /* Check output file for saved checkpoint */
  file1 = fopen((*cfg).ofn, "r");
  if (file1){
      while((ch=fgetc(file1)) != EOF){
          if(ch == '\n'){
              k_start++;
          }
      }
  fclose(file1);
  printf("Backup collected. Starting from k = %i\n", k_start);
  }
  else{
      printf("No backup file. Starting from fmin\n");
      k_start = 0;
  }
  printf ("Checkpointing every %i iterations\n", Nsave);
  printf ("Computing output:  00.0%%");
  fflush (stdout);
  gettimeofday (&tv, NULL);
  start = tv.tv_sec + tv.tv_usec / 1e6;
  now = start;
  print = start;
  
  /* Start calculation of LPSD from saved checkpoint or zero */
  struct hdf5_contents *contents = read_hdf5_file((*cfg).ifn, (*cfg).dataset_name);
  for (k = k_start; k < (*cfg).nspec; k++)
    {
      getDFT2((*data).nffts[k], (*data).bins[k], (*cfg).fsamp, (*cfg).ovlp,
	          &rslt[0], &(*data).avg[k], contents);

      (*data).psd[k] = rslt[0];
      (*data).varpsd[k] = rslt[1];
      (*data).ps[k] = rslt[2];
      (*data).varps[k] = rslt[3];
      gettimeofday (&tv, NULL);
      now = tv.tv_sec + tv.tv_usec / 1e6;
      if (now - print > PSTEP)
	{
	  print = now;
	  progress = (100 * ((double) k)) / ((double) ((*cfg).nspec));
	  printf ("\b\b\b\b\b\b%5.1f%%", progress);
	  fflush (stdout);
	}

      /* If k is a multiple of Nsave then write data to backup file */
      if(k % Nsave  == 0 && k != k_start){
          file1 = fopen((*cfg).ofn, "a");
          for(j=k-Nsave; j<k; j++){
		fprintf(file1, "%e	", (*data).psd[j]);
		fprintf(file1, "%e	", (*data).ps[j]);
		fprintf(file1, "%d	", (*data).avg[j]);
		fprintf(file1, "\n");
          }
          fclose(file1);
      }
      else if(k == (*cfg).nspec - 1){
          file1 = fopen((*cfg).ofn, "a");
          for(j=Nsave*(k/Nsave); j<(*cfg).nspec; j++){
		fprintf(file1, "%e	", (*data).psd[j]);
		fprintf(file1, "%e	", (*data).ps[j]);
		fprintf(file1, "%d	", (*data).avg[j]);
		fprintf(file1, "\n");
          }
          fclose(file1);
      }
    }
  /* finish */
  close_hdf5_contents(contents);
  printf ("\b\b\b\b\b\b  100%%\n");
  fflush (stdout);
  gettimeofday (&tv, NULL);
  printf ("Duration (s)=%5.3f\n\n", tv.tv_sec - start + tv.tv_usec / 1e6);
}


// @brief Calculate FFT on data of length N
// @brief This implementation puts everything in memory, serves as test
// @brief Takes in real data
// @param Custom bin number, necessary for logarithmic frequency spacing
// @brief Memory contents reach 3N in main loop (+ 2N from recursion)
// TODO: implement Bergland's algorithm
// TODO: sin/cos optimisation
void
FFT(double *data_real, double *data_imag, int N,
    double *output_real, double *output_imag)
{
    if (N == 1) {
        output_real[0] = data_real[0];
        output_imag[0] = data_imag[0];
        return;
    }
    int m = N / 2;

    // Separate data in even and odd parts
    double *x_even_real = (double*) xmalloc(m*sizeof(double));
    double *x_even_imag = (double*) xmalloc(m*sizeof(double));
    stride_over_array(data_imag, N, 2, 0, x_even_imag);
    stride_over_array(data_real, N, 2, 0, x_even_real);
    // Calculate FFT over halved arrays
    double *X_even_real = (double*) xmalloc(m*sizeof(double));
    double *X_even_imag = (double*) xmalloc(m*sizeof(double));
    FFT(x_even_real, x_even_imag, m, X_even_real, X_even_imag);
    // Clean up
    xfree(x_even_real);
    xfree(x_even_imag);

    // Repeat for odd part
    double *x_odd_real = (double*) xmalloc(m*sizeof(double));
    double *x_odd_imag = (double*) xmalloc(m*sizeof(double));
    stride_over_array(data_real, N, 2, 1, x_odd_real);
    stride_over_array(data_imag, N, 2, 1, x_odd_imag);
    // Calculate FFT over halved arrays
    double *X_odd_real = (double*) xmalloc(m*sizeof(double));
    double *X_odd_imag = (double*) xmalloc(m*sizeof(double));
    FFT(x_odd_real, x_odd_imag, m, X_odd_real, X_odd_imag);
    // Clean up
    xfree(x_odd_real);
    xfree(x_odd_imag);

    // Calculate exponential term to multiply to X_odd
    double exp_factor = 2.0 * M_PI / ((double) N);
    for (int i = 0; i < m; i++) {
        double y = cos(i*exp_factor);
        double x = -sin(i*exp_factor);
        double b = X_odd_real[i];
        double a = X_odd_imag[i];
        X_odd_real[i] = b*y - a*x;
        X_odd_imag[i] = a*y + b*x;
    }

    // Calculate final answer
    for (int i = 0; i < m; i++) {
        output_real[i] = X_even_real[i] + X_odd_real[i];
        output_imag[i] = X_even_imag[i] + X_odd_imag[i];
        output_real[i+m] = X_even_real[i] - X_odd_real[i];
        output_imag[i+m] = X_even_imag[i] - X_odd_imag[i];
    }

    // Clean up
    xfree(X_even_real);
    xfree(X_odd_real);
    xfree(X_even_imag);
    xfree(X_odd_imag);
}


// @brief Use const. N approximation for a given epsilon
void
calculate_fft_approx (tCFG * cfg, tDATA * data)
{
    // Track time and progress
    struct timeval tv;
    printf ("Computing output:  00.0%%");
    fflush (stdout);
    gettimeofday (&tv, NULL);
    double start = tv.tv_sec + tv.tv_usec / 1e6;
    double now, print, progress;
    print = start;
    now = start;

    // Define variables
    double epsilon = 0.1;  // TODO: pass arg

    // Prepare data file
    struct hdf5_contents *contents = read_hdf5_file((*cfg).ifn, (*cfg).dataset_name);

    register int i;
    int j, j0;
    j = j0 = 0;
    double g = log(cfg->fmax / cfg->fmin);

    while (true) {
        // Get index of the end of the block - the frequency at which the approximation is valid up to epsilon
        j0 = j;
        int Nj0 = get_N_j(j0, cfg->fsamp, cfg->fmin, cfg->fmax, cfg->Jdes);
        j = - (cfg->Jdes - 1.) / g * log(Nj0*(1. - epsilon) * cfg->fmin/cfg->fsamp * (exp(g / (cfg->Jdes - 1.)) - 1.));
        if (j >= cfg->Jdes) break; // TODO: take care of edge case

        // Calculate window
        double *window = (double*) xmalloc(Nj0*sizeof(double));
        makewin(Nj0, window, &winsum, &winsum2, &nenbw);

        // Initialise data arrays
        int Nfft = get_next_power_of_two(Nj0);
        double *data_real = (double*) xmalloc(Nfft*sizeof(double));
        double *data_imag = (double*) xmalloc(Nfft*sizeof(double));
        double *fft_real = (double*) xmalloc(Nfft*sizeof(double));
        double *fft_imag = (double*) xmalloc(Nfft*sizeof(double));
        memset(data_imag, 0, Nfft*sizeof(double));
        for (i = Nj0; i < Nfft; i++) data_real[i] = 0;

        // Loop over segments
        int delta_segment = floor(Nj0 * (1.0 - (double) (cfg->ovlp / 100.)));
        int n_segments = floor(1 + (nread - Nj0) / delta_segment);
        /* Adjust for edge case */
        int tmp = (n_segments - 1)*delta_segment + Nj0;
        if (tmp == nread) n_segments--;
        double total[j - j0];
        memset(total, 0, (j-j0)*sizeof(double));

        register int i_segment, ji;
        for (i_segment = 0; i_segment < n_segments; i_segment++){
            // Take FFT of data x window
            hsize_t offset[1] = {i_segment*delta_segment};
            hsize_t count[1] = {Nj0};
            read_from_dataset(contents, offset, count, data_real);
            for (i = 0; i < Nj0; i++) data_real[i] *= window[i];
            FFT(data_real, data_imag, Nfft, fft_real, fft_imag);
            // Interpolate results
            for (ji = j0; ji < j; ji++) {
                int jfft = floor(Nfft * cfg->fmin/cfg->fsamp * exp(ji*g/(cfg->Jdes - 1.)));
                double x = get_f_j(ji, cfg->fmin, cfg->fmax, cfg->Jdes);
                double y1 = fft_real[jfft]*fft_real[jfft] + fft_imag[jfft]*fft_imag[jfft];
                double y2 = fft_real[jfft+1]*fft_real[jfft+1] + fft_imag[jfft+1]*fft_imag[jfft+1];
                double x1 = cfg->fsamp / Nfft * jfft;
                double x2 = cfg->fsamp / Nfft * (jfft+1);
                total[ji - j0] += (y1*(x2 - x) - y2*(x1 - x)) / (x2 - x1);
            }
        }
        // Normalise results and add to data->psd/data->ps
        for (ji = 0; ji < j - j0; ji++) {
            data->psd[ji+j0] = total[ji] * 2. / (n_segments * cfg->fsamp * winsum2);
            data->ps[ji+j0] = total[ji] * 2 / (n_segments * winsum*winsum);
        }

        // Progress tracking
        progress = 100. * (double) j / cfg->Jdes;
        printf ("\b\b\b\b\b\b%5.1f%%", progress);
        fflush (stdout);

        // Clean up
        xfree(data_real);
        xfree(data_imag);
        xfree(fft_real);
        xfree(fft_imag);
    }
    // Finish
    gettimeofday (&tv, NULL);
    printf ("\b\b\b\b\b\b  100%%\n");
    printf ("Duration (s)=%5.3f\n\n", tv.tv_sec - start + tv.tv_usec / 1e6);

    // Clean up
    close_hdf5_contents(contents);
}


/*
	works on cfg, data structures of the calling program
*/
void
calculateSpectrum (tCFG * cfg, tDATA * data)
{
  nread = floor (((*cfg).tmax - (*cfg).tmin) * (*cfg).fsamp + 1);

  calc_params (cfg, data);
  if ((*cfg).METHOD == 0) calculate_lpsd (cfg, data);
  else if ((*cfg).METHOD == 1) calculate_fft_approx (cfg, data);
  else gerror("Method not implemented.");
}
