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
double get_mean(int* values, int N) {
    double _sum = 0;
    register int i;
    for (i = 0; i < N; i++) {
        _sum += values[i];
    }
    return _sum / N;
}


// @brief recursive function to count set bits
int countSetBits(int n)
{
    // base case
    if (n == 0)
        return 0;
    else
        // if last bit set add 1 else add 0
        return (n & 1) + countSetBits(n >> 1);
}


bool isPowerOfTwo(int n)
{
    return countSetBits(n) == 1;
}


static void
getDFT2 (int nfft, double bin, double fsamp, double ovlp, int LR, double *rslt,
         int *avg, struct hdf5_contents *contents)
{
  /* Configure variables for DFT */
//  int max_samples_in_memory = 5*6577770;  // Around 500 MB //TODO: this shouldn't be hard-coded!
  int max_samples_in_memory = 512;  // tmp
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
	      (*cfg).LR, &rslt[0], &(*data).avg[k], contents);

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


double*
stride_over_array(double *data, int N, int stride, int offset, double *output)
{
    for (int i = offset; i < N; i += stride) *(output++) = data[i];
}

// @brief Calculate FFT on data of length N
// @brief This implementation puts everything in memory, serves as test
// @brief Takes in real data
// @param Custom bin number, necessary for logarithmic frequency spacing
// TODO: implement Bergland's algorithm
// TODO: sin/cos optimisation
void
FFT(double *data_real, double *data_imag, int N, double bin,
    double *output_real, double *output_imag)
{
    if (N == 1) {
        output_real[0] = data_real[0];
        output_imag[0] = data_imag[0];
        return;
    }
    int m = N / 2;

    // Separate data in even and odd parts
    double x_even_real[m], x_odd_real[m], x_even_imag[m], x_odd_imag[m];
    stride_over_array(data_real, N, 2, 0, x_even_real);
    stride_over_array(data_real, N, 2, 1, x_odd_real);
    stride_over_array(data_imag, N, 2, 0, x_even_imag);
    stride_over_array(data_imag, N, 2, 1, x_odd_imag);

    // Calculate FFT over halved arrays
    double X_even_real[m], X_even_imag[m], X_odd_real[m], X_odd_imag[m];
    FFT(x_even_real, x_even_imag, m, bin, X_even_real, X_even_imag);
    FFT(x_odd_real, x_odd_imag, m, bin, X_odd_real, X_odd_imag);

    // Calculate exponential term to multiply to X_odd
    double exp_factor = 2.0 * M_PI * bin / ((double) N);
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
//    printf("test\n\t%d, %e, %e\n", m, output_imag[1], output_real[1]);
//    printf("0:\t%d, %e, %e\n", m, X_odd_real[0], X_even_real[0]);
//    printf("\t%d, %e, %e\n", m, X_odd_imag[0], X_even_imag[0]);
//    printf("x:\t%d, %e, %e\n", m, x_odd_imag[0], x_even_imag[0]);
//    printf("x:\t%d, %e, %e\n", m, x_odd_real[0], x_even_real[0]);
//    printf("1:\t%d, %e, %e\n", m, X_odd_real[1], X_even_real[1]);
//    printf("\t%d, %e, %e\n", m, X_odd_imag[1], X_even_imag[1]);
//    printf("x:\t%d, %e, %e\n", m, x_odd_imag[1], x_even_imag[1]);
//    printf("x:\t%d, %e, %e\n", m, x_odd_real[1], x_even_real[1]);
}

// @brief Same as FFT, but use fftw3 implementation instead, for speed.

// @brief Use FFT approximation to calculate LPSD much faster, but at a certain cost of precision
// @brief For now, just run over all frequency bins
// @brief In the future, this can be parallelized by splitting which segments are calculated?
// @brief though it would require a final job to average etc.
void
calculate_fft_approx (tCFG * cfg, tDATA * data)
{
    // User output
    struct timeval tv;
    int Nsave = (*cfg).nspec / 100;
    printf ("Checkpointing every %i iterations\n", Nsave);
    printf ("Computing output:  00.0%%");
    fflush (stdout);
    gettimeofday (&tv, NULL);
    double start = tv.tv_sec + tv.tv_usec / 1e6;
    double now, print, progress;
    print = now = start;

    // Start LPSD calculation
    struct hdf5_contents *contents = read_hdf5_file((*cfg).ifn, (*cfg).dataset_name);

    // For reference, calculate FFT for each segment
    // Assume that all segments have the same length: mean of nffts
    int n_frequency_bins = cfg->nspec;
    int segment_length = round(get_mean(data->nffts, n_frequency_bins));
    int delta_segment = floor(segment_length * (1.0 - (double) (cfg->ovlp / 100.)));
    int n_segments = floor(1 + (nread - segment_length ) / delta_segment);
    /* Adjust for edge case */
    int tmp = (n_segments - 1)*delta_segment + segment_length;
    if (tmp == nread) n_segments--;
    // Comment: bin is constant in this version of the code, which is not entirely correct
    // This will have to do for now
    double bin = data->bins[0];

    // Prepare zero-padding for FFT
    int fft_length;
    if (isPowerOfTwo(segment_length)) fft_length = segment_length;
    else fft_length = pow(2, (int) log2(segment_length) + 1);

    double *strain_data_segment = (double*) xmalloc(fft_length*sizeof(double));
    double *window = (double*) xmalloc(2*fft_length*sizeof(double));
    // Set remainder to zero
    register int i;
    for (i = segment_length; i < fft_length; i++) {
        strain_data_segment[i] = 0;
        window[2*i] = window[2*i+1] = 0;
    }

    // Calculate window
    makewin(segment_length, window, &winsum, &winsum2, &nenbw);

    // Loop over segments
    hsize_t count[1] = {segment_length};
    double total[cfg->nspec];
    memset(total, 0, cfg->nspec*sizeof(double));
    for (int i_segment = 0; i_segment < n_segments; i_segment++) {
        // Read segment
        hsize_t offset[1] = {i_segment * delta_segment};
        read_from_dataset(contents, offset, count, strain_data_segment);

        // FFT calculation
        double zeros[fft_length];  // Imaginary part of data
        memset(zeros, 0, fft_length*sizeof(double));
        double output_real[fft_length], output_imag[fft_length];
        FFT(strain_data_segment, zeros, fft_length, bin, output_real, output_imag);

        // Save results
        for (i = 0; i < cfg->nspec; i++) {
            double _sum = output_real[i]*output_real[i] + output_imag[i]*output_imag[i];
            total[i] += _sum;
        }

        // Track progress
//        now = tv.tv_sec + tv.tv_usec / 1e6;
//        print = now;
//        progress = (100 * ((double) i_segment)) / ((double) n_segments);
//        printf ("\b\b\b\b\b\b%5.1f%%", progress);
//        fflush (stdout);
    }
    // Average over segments
    for (i = 0; i < cfg->nspec; i++){
        data->psd[i] = total[i] * 2. / (n_segments * cfg->fsamp * winsum2);
        data->ps[i] = data->psd[i] * (cfg->fsamp * winsum2) / (winsum * winsum);
        data->varpsd[i] = data->varps[i] = 0;
        data->avg[i] = n_segments;
    }


    /* finish */
    fflush (stdout);
    gettimeofday (&tv, NULL);
    printf ("\b\b\b\b\b\b  100%%\n");
    printf ("Duration (s)=%5.3f\n\n", tv.tv_sec - start + tv.tv_usec / 1e6);

    /* clean up */
    close_hdf5_contents(contents);
    xfree(window);
    xfree(strain_data_segment);
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
