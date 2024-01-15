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
static long int nread;
static double winsum;
static double winsum2;
static double nenbw;		/* normalized equivalent noise bandwidth */
static double *dwin;		/* pointer to window function for FFT */

/********************************************************************************
 * 	functions
 ********************************************************************************/


// Interpolate between x12, y12 points to x
double
interpolate(double x, double x1, double x2, double y1, double y2) {
    return (y1*(x2 - x) - y2*(x1 - x)) / (x2 - x1);
}


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


// @brief implementation of pow() that works with long ints as base and exponent
long int ld_pow(long int base, long int exponent) {
    long int result = 1;
    while (exponent > 0) {
        result *= base;
        exponent--;
    }
    return result;
}


// @brief implementation of int(log2()) that works with long ints
int ld_log2(unsigned long int n) {
    int count = -1;
    while (n > 0) {
        n >>= 1;
        count++;
    }
    return count;
}


// @brief recursive function to count set bits
int
count_set_bits (long int n)
{
    // base case
    if (n == 0)
        return 0;
    else
        // if last bit set add 1 else add 0
        return (n & 1) + count_set_bits(n >> 1);
}


unsigned long int
get_next_power_of_two (unsigned long int n)
{
    unsigned long int output;
    if (!(count_set_bits(n) == 1 || n == 0))
      output = (unsigned long int) ld_pow(2, (long int) (ld_log2(n) + 1));
    return output;
}


void
stride_over_array (double *data, int N, int stride, int offset, double *output)
{
    for (int i = offset; i < N; i += stride) *(output++) = data[i];
}


// Get the segment length as a function of the frequency bin j
// Rounded to nearest integer.
// TODO: replace with call to nffts?
unsigned long int
get_N_j (double j, double fsamp, double fmin, double fmax, double Jdes) {
    double g = log(fmax) - log(fmin);  // TODO: could consider making g part of cfg
    // This exp turn is in overflow danger
    return round (fsamp/fmin * exp(-j*g / (Jdes - 1.)) / (exp(g / (Jdes - 1.)) - 1.));
}

// Get the frequency of bin j
// TODO: replace with call to fspec?
double
get_f_j (double j, double fmin, double fmax, double Jdes) {
    double g = log(fmax) - log(fmin);  // TODO: could consider making g part of cfg
    return fmin*exp(j*g / (Jdes - 1.));
}

void
fill_ordered_coefficients(int n, int *coefficients) {
    coefficients[0] = 0;
    coefficients[1] = 1;
    if (n == 1) return;
    for (int m = 2; m <= n; m++) {
        int two_to_m = pow(2, m);
        int two_to_m_minus_one = pow(2, m-1);
        int tmp[two_to_m];
        for (int i = 0; i < two_to_m_minus_one; i++) {
            tmp[2*i] = coefficients[i];
            tmp[2*i+1] = coefficients[i] + two_to_m_minus_one;
        }
        for (int i = 0; i < two_to_m; i++) coefficients[i] = tmp[i];
    }
}


static void
getDFT2 (long int nfft, double bin, double fsamp, double ovlp, double *rslt,
         int *avg, struct hdf5_contents *contents, int max_samples_in_memory)
{
  /* Configure variables for DFT */
  if (max_samples_in_memory > nfft) max_samples_in_memory = nfft; // Don't allocate more than you need
  // TODO: getDFT2 will stop working if there are more than 2 segments at j0=0 !!

  /* Allocate data and window memory segments */
  double *strain_data_segment = (double*) xmalloc(max_samples_in_memory * sizeof(double));
  double *window = (double*) xmalloc(2*max_samples_in_memory * sizeof(double));
  assert(window != 0 && strain_data_segment != 0);

  //////////////////////////////////////////////////
  /* Calculate DFT over separate memory windows */
  long int window_offset;
  int count;
  int memory_unit_index = 0;
  long int remaining_samples = nfft;
  int nsum = floor(1+(nread - nfft) / floor(nfft * (1.0 - (double) (ovlp / 100.))));
  long int tmp = (nsum-1)*floor(nfft * (1.0 - (double) (ovlp / 100.)))+nfft;
  if (tmp == nread) nsum--;  /* Adjust for edge case */
  printf("nsum: %i\n", nsum);

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
    register unsigned int i;
    unsigned long int start = 0;
    register unsigned int _nsum = 0;
    hsize_t data_count[1] = {count};
    hsize_t data_rank = 1;
    while (start + nfft < nread)
    {
      // Load data
      hsize_t data_offset[1] = {start + window_offset};
      read_from_dataset(contents, data_offset, data_count, data_rank, data_count, strain_data_segment);

      // Calculate DFT
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
  long int i, i0, ndft;

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
  int max_samples_in_memory = pow(2, cfg->n_max_mem);
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
  struct hdf5_contents contents;
  read_hdf5_file(&contents, (*cfg).ifn, (*cfg).dataset_name);
  for (k = k_start; k < (*cfg).nspec; k++)
    {
      getDFT2((*data).nffts[k], (*data).bins[k], (*cfg).fsamp, (*cfg).ovlp,
              &rslt[0], &(*data).avg[k], &contents, max_samples_in_memory);

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
  close_hdf5_contents(&contents);
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
FFT(double *data_real, double *data_imag, unsigned int N,
    double *output_real, double *output_imag)
{
    if (N == 1) {
        output_real[0] = data_real[0];
        output_imag[0] = data_imag[0];
        return;
    }
    unsigned int m = N / 2;

    // Separate even part in real/imaginary
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


// Perform an FFT while controlling how much gets in memory by manually calculating the
// top layers of the pyramid over sums
void
FFT_control_memory(unsigned long int Nj0, unsigned long int Nfft, unsigned int Nmax,
                   unsigned long int segment_offset, struct hdf5_contents *contents,
                   struct hdf5_contents *window_contents, struct hdf5_contents *_contents)
{
    // Determine manual recursion depth
    // Nfft and Nmax must be powers of two!!
    int n_depth = ld_log2(Nfft) - ld_log2(Nmax);

    // Get 2^n_depth data samples, then iteratively work down to n = 1
    unsigned int two_to_n_depth = (unsigned int) ld_pow(2, n_depth);  // use long for compatibility with ld_pow, but this number should be small enough
    unsigned int Nj0_over_two_n_depth = Nj0 / two_to_n_depth;  // equivalent to floor(..)
    int ordered_coefficients[two_to_n_depth];
    fill_ordered_coefficients(n_depth, ordered_coefficients);

    // Approx (5 * 16 * Nmax) bits in memory
    double *data_subset_real = (double*)malloc(Nmax*sizeof(double));
    double *data_subset_imag = (double*)malloc(Nmax*sizeof(double));
    memset(data_subset_imag, 0, Nmax*sizeof(double));
    double *fft_output_real = (double*)malloc(Nmax*sizeof(double));
    double *fft_output_imag = (double*)malloc(Nmax*sizeof(double));
    double *window_subset = (double*)malloc((Nj0_over_two_n_depth+1)*sizeof(double));

    // Perform FFTs on bottom layer of pyramid and save results to temporary file
    for (unsigned int i = 0; i < two_to_n_depth; i++) {
        // Read data
        hsize_t offset[1] = {ordered_coefficients[i] + segment_offset};
        unsigned int Ndata = ordered_coefficients[i] + (unsigned long int)Nj0_over_two_n_depth*two_to_n_depth < Nj0 ?
            Nj0_over_two_n_depth + 1 : Nj0_over_two_n_depth;
        hsize_t count[1] = {Ndata};  // Note. Ndata will always be compatible with int by design
        hsize_t stride[1] = {two_to_n_depth};
        hsize_t rank = 1;
        read_from_dataset_stride(contents, offset, count, stride, rank, count, data_subset_real);

        // Zero-pad data
        for (unsigned int j = Ndata; j < Nmax; j++) data_subset_real[j] = 0;

        // Read window & apply to data (piecewise multiply)
        hsize_t window_offset[1] = {ordered_coefficients[i]};
        read_from_dataset_stride(window_contents, window_offset, count, stride, rank, count, window_subset);
        for (unsigned int j = 0; j < Ndata; j++) data_subset_real[j] *= window_subset[j];

        // Take FFT
        FFT(data_subset_real, data_subset_imag, Nmax, fft_output_real, fft_output_imag);

        // Save real part to file
        hsize_t _offset[2] = {0, i*(unsigned long int)Nmax};
        hsize_t _count[2] = {1, Nmax};
        hsize_t _data_rank = 1;
        hsize_t _data_count[1] = {Nmax};
        write_to_hdf5(_contents, fft_output_real, _offset, _count, _data_rank, _data_count);
        // Save imaginary part
        _offset[0] = 1;
        write_to_hdf5(_contents, fft_output_imag, _offset, _count, _data_rank, _data_count);
        // Note: if I saved real/imag in one 2D array instead of two arrays,
        // I would only need one call to write_to_hdf5 in this loop. Could save time?
    }
    // Clean-up
    free(data_subset_real);
    free(data_subset_imag);
    free(window_subset);
    free(fft_output_real);
    free(fft_output_imag);

    // TODO: don't need to write the last iteration of the pyramid to file as I could work with it here directly, small speed-up
    // Put 5 * 16 * Nmax bits in memory
    double *even_terms_real = (double*)malloc(Nmax*sizeof(double));
    double *even_terms_imag = (double*)malloc(Nmax*sizeof(double));
    double *odd_terms_real = (double*)malloc(Nmax*sizeof(double));
    double *odd_terms_imag = (double*)malloc(Nmax*sizeof(double));
    double *write_vector = (double*)malloc(Nmax*sizeof(double));
    // Now loop over the rest of the pyramid
    while (n_depth > 0) {
        // Iterate n_depth
        n_depth--;
        two_to_n_depth = (unsigned int) ld_pow(2, n_depth);
        Nj0_over_two_n_depth = Nj0 / two_to_n_depth;
        unsigned long int Nfft_over_two_n_depth = round(Nfft / two_to_n_depth);
        // Number of memory units in lower-level pyramid segment
        int n_mem_units = pow(2, (int)round(ld_log2(Nfft) - ld_log2(Nmax) - n_depth - 1));

        // Loop over segments at this pyramid level
        for (unsigned int i_pyramid = 0; i_pyramid < two_to_n_depth; i_pyramid++) {
            // Loop over memory units (of length Nmax) in one lower-level segment
            for (unsigned int j = 0; j < n_mem_units; j++) {
                // Load even terms
                hsize_t offset[2] = {0, (j + 2*i_pyramid*n_mem_units)*(unsigned long int)Nmax};
                hsize_t count[2] = {1, Nmax};
                hsize_t data_rank = 1;
                hsize_t data_count[1] = {Nmax};
                read_from_dataset(_contents, offset, count, data_rank, data_count, even_terms_real);
                offset[0] = 1;
                read_from_dataset(_contents, offset, count, data_rank, data_count, even_terms_imag);

                // Load odd terms
                offset[1] += n_mem_units*(unsigned long int)Nmax;
                read_from_dataset(_contents, offset, count, data_rank, data_count, odd_terms_imag);
                offset[0] = 0;
                read_from_dataset(_contents, offset, count, data_rank, data_count, odd_terms_real);

                // Piecewise (complex) multiply odd terms with exp term
                double exp_factor = 2.0 * M_PI / ((double) Nfft_over_two_n_depth);
                for (unsigned int k = 0; k < Nmax; k++) {
                    double y = cos((j*(double)Nmax+k)*exp_factor);
                    double x = -sin((j*(double)Nmax+k)*exp_factor);
                    double a = odd_terms_imag[k];
                    double b = odd_terms_real[k];
                    odd_terms_real[k] = b*y - a*x;
                    odd_terms_imag[k] = a*y + b*x;
                }

                // Combine left side
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_real[k] + odd_terms_real[k];
                hsize_t offset_left[2] = {0, (j + 2*i_pyramid*n_mem_units)*(unsigned long int)Nmax};
                write_to_hdf5(_contents, write_vector, offset_left, count, data_rank, data_count);
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_imag[k] + odd_terms_imag[k];
                offset_left[0] = 1;
                write_to_hdf5(_contents, write_vector, offset_left, count, data_rank, data_count);

                // Combine right side
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_real[k] - odd_terms_real[k];
                hsize_t offset_right[2] = {0, (j + (2*i_pyramid + 1)*n_mem_units)*(unsigned long int)Nmax};
                write_to_hdf5(_contents, write_vector, offset_right, count, data_rank, data_count);
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_imag[k] - odd_terms_imag[k];
                offset_right[0] = 1;
                write_to_hdf5(_contents, write_vector, offset_right, count, data_rank, data_count);
            }
        }
    }
    // Clean up
    free(even_terms_real);
    free(even_terms_imag);
    free(odd_terms_real);
    free(odd_terms_imag);
    free(write_vector);
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
    double epsilon = cfg->epsilon / 100.;
    double g = log(cfg->fmax / cfg->fmin);
    unsigned int max_samples_in_memory = pow(2, cfg->n_max_mem);

    // Prepare data file
    struct hdf5_contents contents;
    read_hdf5_file(&contents, (*cfg).ifn, (*cfg).dataset_name);

    // Loop over blocks
    register int i;
    unsigned int j, j0;  // indices over frequency space
    j = j0 = 0;
    while (j < cfg->nspec - 1) {
        // Get index of the end of the block - the frequency at which the approximation is valid up to epsilon
        // Block starts at j0
        j0 = j;
        // Nj0: length of segment
        unsigned long int Nj0 = get_N_j(j0, cfg->fsamp, cfg->fmin, cfg->fmax, cfg->Jdes);
        // Block ends at j
        j = - ((double)cfg->Jdes - 1.) / g * log(Nj0*(1. - epsilon) * cfg->fmin/cfg->fsamp * (exp(g / ((double)cfg->Jdes - 1.)) - 1.));
        if (j >= cfg->nspec) j = cfg->nspec-1;

        // Only run this segment if it overlaps with fmin_fft and fmax_fft coverage
        if (cfg->fmax_fft > 0 && data->fspec[j0] >= cfg->fmax_fft)
        	continue;

		if (cfg->fmin_fft > 0 && data->fspec[j] <= cfg->fmin_fft)
			continue;

        // Prepare segment loop
        unsigned long int delta_segment = floor(Nj0 * (1.0 - (double) (cfg->ovlp / 100.)));
        unsigned int n_segments = floor(1 + (nread - Nj0) / delta_segment);
        /* Adjust for edge case */
        long int tmp = (n_segments - 1)*delta_segment + Nj0;
        if (tmp == nread) n_segments--;

		// Allocate arrays used to store the results in between
		double *total = (double*) xmalloc((j - j0)*sizeof(double));
		double *total_real = (double*) xmalloc((j - j0)*sizeof(double));
		double *total_imag = (double*) xmalloc((j - j0)*sizeof(double));
		memset(total, 0, (j - j0)*sizeof(double));
		memset(total_real, 0, (j - j0)*sizeof(double));
		memset(total_imag, 0, (j - j0)*sizeof(double));

		// Allocate arrays used to store the raw results
		double *total_raw = (double*) xmalloc((j - j0)*sizeof(double));
		double *total_raw_real = (double*) xmalloc((j - j0)*sizeof(double));
		double *total_raw_imag = (double*) xmalloc((j - j0)*sizeof(double));
		memset(total_raw, 0, (j - j0)*sizeof(double));
		memset(total_raw_real, 0, (j - j0)*sizeof(double));
		memset(total_raw_imag, 0, (j - j0)*sizeof(double));

		// Prepare FFT
		// Whatever the value of max is, make it less than 2^31 or ints will break
		unsigned long int Nfft = get_next_power_of_two(Nj0);
        // Relevant frequency range in full fft space
        unsigned int jfft_min = floor(Nfft * cfg->fmin/cfg->fsamp * exp(j0*g/(cfg->Jdes - 1.)));
        unsigned int jfft_max = ceil(Nfft * cfg->fmin/cfg->fsamp * exp(j*g/(cfg->Jdes - 1.)));

        double *data_real, *data_imag, *fft_real, *fft_imag, *window;
        data_real = data_imag = fft_real = fft_imag = window = NULL;
        struct hdf5_contents _contents, window_contents;
        struct hdf5_contents *_contents_ptr = NULL, *window_contents_ptr = NULL;
        // This if/else statement allocates variables for the steps to come
        if (Nfft <= max_samples_in_memory) {
            // For normal FFT
            // Initialiase data/window arrays
            data_real = (double*) xmalloc(Nfft*sizeof(double));
            data_imag = (double*) xmalloc(Nfft*sizeof(double));
            fft_real = (double*) xmalloc(Nfft*sizeof(double));
            fft_imag = (double*) xmalloc(Nfft*sizeof(double));
            memset(data_imag, 0, Nfft*sizeof(double));
            for (i = Nj0; i < Nfft; i++) data_real[i] = 0;

            // Calculate window
            window = (double*) xmalloc(Nj0*sizeof(double));
            makewin(Nj0, window, &winsum, &winsum2, &nenbw);
        } else {
            // For memory-controlled FFT
            // Open temporary hdf5 file to temporarily store information to disk in the loop
            hsize_t rank = 2;  // real + imaginary
            hsize_t dims[2] = {2, Nfft};
            _contents_ptr = &_contents;
            open_hdf5_file(_contents_ptr, "tmp.h5", "fft_contents", rank, dims);

            // Initialise fft output, but only in relevant frequency range
            fft_real = (double*) xmalloc((jfft_max - jfft_min)*sizeof(double));
            fft_imag = (double*) xmalloc((jfft_max - jfft_min)*sizeof(double));

            // Calculate window and put it in temporary file
            hsize_t window_rank = 1;
            hsize_t window_dims[1] = {Nj0};
            window_contents_ptr = &window_contents;
            open_hdf5_file(window_contents_ptr, "window.h5", "window", window_rank, window_dims);

            // Loop over Nmax segments to calculate window without exceeding max memory
            window = (double*) xmalloc(max_samples_in_memory*sizeof(double));
            unsigned long int remaining_samples = Nj0;
            unsigned int memory_unit_index = 0;
            unsigned int iteration_samples;
            while (remaining_samples > 0) {
                // Calculate window
                if (remaining_samples > max_samples_in_memory) iteration_samples = max_samples_in_memory;
                else iteration_samples = remaining_samples;
                makewin_indexed(Nj0, memory_unit_index*max_samples_in_memory,
                                iteration_samples, window,
                                &winsum, &winsum2, &nenbw, memory_unit_index == 0);

                // Save to file
                hsize_t offset[1] = {memory_unit_index*max_samples_in_memory};
                hsize_t count[1] = {iteration_samples};
                write_to_hdf5(window_contents_ptr, window, offset, count, window_rank, count);
                // Book-keeping
                remaining_samples -= iteration_samples;
                memory_unit_index++;
            }
            xfree(window);
            window = NULL;
        }

        register int i_segment, ji;
        // Loop over segments - this is the actual calculation step
        for (i_segment = 0; i_segment < n_segments; i_segment++) {
            int index_shift = 0;
            if (Nfft <= max_samples_in_memory) {
                // Run normal FFT
                hsize_t offset[1] = {i_segment*delta_segment};
                hsize_t count[1] = {Nj0};
                hsize_t data_rank = 1;
                hsize_t data_count[1] = {count[0]};
                read_from_dataset(&contents, offset, count, data_rank, data_count, data_real);
                for (i = 0; i < Nj0; i++) data_real[i] *= window[i];
                FFT(data_real, data_imag, (int)Nfft, fft_real, fft_imag);
            } else {
                // Run memory-controlled FFT
                FFT_control_memory(Nj0, Nfft, max_samples_in_memory, i_segment*delta_segment,
                                   &contents, &window_contents, &_contents);
                // Load frequency domain results between j0 and j
                hsize_t count[2] = {1, jfft_max - jfft_min};
                hsize_t offset[2] = {0, jfft_min};
                hsize_t data_rank = 1;
                hsize_t data_count[1] = {count[1]};
                read_from_dataset(&_contents, offset, count, data_rank, data_count, fft_real);
                offset[0] = 1;
                read_from_dataset(&_contents, offset, count, data_rank, data_count, fft_imag);
                index_shift = jfft_min;
            }
            // Interpolate results (linear)
            for (ji = j0; ji < j; ji++) {
                int jfft = floor(Nfft * cfg->fmin/cfg->fsamp * exp(ji*g/(cfg->Jdes - 1.)));
                double x = get_f_j(ji, cfg->fmin, cfg->fmax, cfg->Jdes);
                double x1 = cfg->fsamp / Nfft * jfft;
                double x2 = cfg->fsamp / Nfft * (jfft+1);

                // Real part
                double y1 = fft_real[jfft-index_shift];
                double y2 = fft_real[jfft-index_shift+1];
                total_real[ji - j0] += interpolate(x, x1, x2, y1, y2);

                // Imaginary part
                double z1 = fft_imag[jfft-index_shift];
                double z2 = fft_imag[jfft-index_shift+1];
                total_imag[ji - j0] += interpolate(x, x1, x2, z1, z2);

                // PSD
                double psd1 = y1*y1 + z1*z1;
                double psd2 = y2*y2 + z2*z2;
                total[ji - j0] += interpolate(x, x1, x2, psd1, psd2);

                // Raw FFT results
                total_raw_real[ji - j0] += y1;
                total_raw_imag[ji - j0] += z1;
                total_raw[ji - j0] += psd1;
            }
        }
        // Normalise results and add to data->psd and data->ps
        double norm_psd = 2. / (n_segments * cfg->fsamp * winsum2);
    	double norm_lin = sqrt (norm_psd);
        double norm_ps = 2 / (n_segments * winsum*winsum);
        for (ji = 0; ji < j - j0; ji++) {
            data->psd[ji+j0] = total[ji] * norm_psd;
            data->ps[ji+j0] = total[ji] * norm_ps;
            data->avg[ji+j0] = n_segments;
            data->psd_real[ji+j0] = total_real[ji] * norm_lin;
            data->psd_imag[ji+j0] = total_imag[ji] * norm_lin;

            // Raw results
            data->psd_raw[ji+j0] = total_raw[ji] * norm_psd;
            data->psd_raw_real[ji+j0] = total_raw_real[ji] * norm_lin;
            data->psd_raw_imag[ji+j0] = total_raw_imag[ji] * norm_lin;
        }

        // Progress tracking
        progress = 100. * (double) j / cfg->Jdes;
        printf ("\b\b\b\b\b\b%5.1f%%", progress);
        fflush (stdout);

        // Clean-up
        if (_contents_ptr) close_hdf5_contents(_contents_ptr);
        if (window_contents_ptr) close_hdf5_contents(window_contents_ptr);
        xfree(total);
        xfree(total_real);
        xfree(total_imag);
        xfree(fft_real);
        xfree(fft_imag);
        xfree(total_raw);
        xfree(total_raw_real);
        xfree(total_raw_imag);
        if (data_real) xfree(data_real);
        if (data_imag) xfree(data_imag);
        if (window) xfree(window);
    }
    /* finish */
    close_hdf5_contents(&contents);
    printf ("\b\b\b\b\b\b  100%%\n");
    fflush (stdout);
    gettimeofday (&tv, NULL);
    printf ("Duration (s)=%5.3f\n\n", tv.tv_sec - start + tv.tv_usec / 1e6);
}

/*
    works on cfg, data structures of the calling program
*/
void
calculateSpectrum (tCFG * cfg, tDATA * data)
{
  // Before running the analysis, check if the output file can be opened.
  FILE *ofp = fopen((*cfg).ofn, "w");
  if (ofp == NULL)
    gerror1("Error opening output file.. Aborting.", (*cfg).ofn);
  fclose(ofp);
  ofp = NULL;

  // Make sure passed variables are ok
  if (cfg->n_max_mem >= 31 || cfg->n_max_mem <= 0) gerror("n_max_mem should be smaller than between 1 and 31");
  if ((cfg->fmin_fft > 0 && cfg->fmax_fft > 0) && cfg->fmin_fft >= cfg->fmax_fft) gerror("fmin_fft should be smaller than fmax_fft!");

  // Run the analysis
  nread = floor (((*cfg).tmax - (*cfg).tmin) * (*cfg).fsamp + 1);

  calc_params (cfg, data);
  if ((*cfg).METHOD == 0) calculate_lpsd (cfg, data);
  else if ((*cfg).METHOD == 1) calculate_fft_approx (cfg, data);
  else gerror("Method not implemented.");
}

