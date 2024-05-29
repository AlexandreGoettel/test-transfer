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
#include "fft.h"

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

static void
getDFT2 (long int nfft, double bin, double fsamp, double ovlp, double *rslt,
         int *avg, struct hdf5_contents *contents, int max_samples_in_memory)
{
  /* Configure variables for DFT */
  if (max_samples_in_memory > nfft) max_samples_in_memory = nfft; // Don't allocate more than you need

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

  double dft_results[2*nsum];  /* Real and imaginary parts of DFTs */
  memset(dft_results, 0, 2*nsum*sizeof(double));

  register unsigned int i;
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
  long int k;			/* 0..nspec */
  long int k_start = 0;		/* N. lines in save file. Post fail start point */
  long int j; 			/* Iteration variables for checkpointing data */
  long int Nsave = (*cfg).nspec / 100; /* Frequency of data checkpointing */
  if (Nsave < 1) Nsave = 1;
  char ch;			/* For scanning through checkpointing file */
  int max_samples_in_memory = pow(2, cfg->n_max_mem);
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
  printf("Backup collected. Starting from k = %li\n", k_start);
  }
  else{
      printf("No backup file. Starting from fmin\n");
      k_start = 0;
  }
  printf ("Checkpointing every %ld iterations\n", Nsave);
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
      progress = (100 * ((double) k) + 1) / ((double) ((*cfg).nspec));
      printf ("\b\b\b\b\b\b%5.1f%%", progress);
      fflush (stdout);
    }

      /* If k is a multiple of Nsave then write data to backup file */
      if(k % Nsave == 0 && k != k_start){
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

// @brief Use constant Q approximation
// @brief This will only work when using the Kaiser window.
void
calculate_constQ_approx (tCFG *cfg, tDATA *data)
{
	// Prepare data file
	struct hdf5_contents contents;
	read_hdf5_file(&contents, (*cfg).ifn, (*cfg).dataset_name);

	// ###### START ANALYSIS ###### //
	double g = log(cfg->fmax / cfg->fmin);
	int m = round(1. / (exp(g / (cfg->Jdes - 1.)) - 1.));  // m as an integer!
	unsigned int max_samples_in_memory = pow(2, cfg->n_max_mem);
	// Make sure Nj0 is even
	unsigned long int Nj0 = round(cfg->fsamp*m/cfg->fmin);
	Nj0 = (Nj0 % 2) ? Nj0 - 1 : Nj0;

	// Calculate window normalisation proportionality constant
	printf("Calculating window normalisation constant..\n");
	double *window = (double*) xmalloc(max_samples_in_memory * sizeof(double));
	unsigned long int remaining_samples = Nj0;
	unsigned int memory_unit_index = 0;
	unsigned int iteration_samples;

	// Calculate window normalisation
	while (remaining_samples > 0) {
		if (remaining_samples > max_samples_in_memory) iteration_samples = max_samples_in_memory;
		else iteration_samples = remaining_samples;
		makewin_indexed(Nj0, memory_unit_index*max_samples_in_memory, iteration_samples, window,
		                &winsum, &winsum2, &nenbw, memory_unit_index == 0);
		// Book-keeping
		remaining_samples -= iteration_samples;
		memory_unit_index++;
	}
	xfree(window);
	double norm_propto_factor = winsum2 / (double) Nj0;

	// Prepare memory protection
	struct hdf5_contents _contents;
	struct hdf5_contents *_contents_ptr = NULL;
	double *data_real, *data_imag, *fft_real, *fft_imag;
	data_real = data_imag = fft_real = fft_imag = NULL;

	// FFT over the (whole!) data
	unsigned long int Nfft = get_next_power_of_two(nread);
	if (Nfft > max_samples_in_memory) {
		hsize_t rank = 2;
		hsize_t dims[2] = {2, Nfft};
		_contents_ptr = &_contents;
		open_hdf5_file(_contents_ptr, "tmp.h5", "fft_contents", rank, dims);
		FFT_control_memory(nread-1, Nfft, max_samples_in_memory, 0,
		                   &contents, NULL, _contents_ptr);
	} else {
		data_real = (double*) xmalloc(Nfft * sizeof(double));
		data_imag = (double*) xmalloc(Nfft * sizeof(double));
		fft_real = (double*) xmalloc(Nfft * sizeof(double));
		fft_imag = (double*) xmalloc(Nfft * sizeof(double));
		memset(data_imag, 0, Nfft * sizeof(double));
		for (int i = nread; i < Nfft; i++) data_real[i] = 0;

		// Read and FFT data
		hsize_t offset[1] = {0};
		hsize_t count[1] = {nread-1};
		hsize_t data_rank = 1;
		hsize_t data_count[1] = {count[0]};
		read_from_dataset(&contents, offset, count, data_rank, data_count, data_real);
		FFT(data_real, data_imag, (int)Nfft, fft_real, fft_imag);
	}

	// Track time and progress
    struct timeval tv;
    printf("Computing output:  00.0%%");
    fflush(stdout);
    gettimeofday(&tv, NULL);
    double start = tv.tv_sec + tv.tv_usec / 1e6;
    double now, print, progress;
    print = now = start;

    // Loop over frequencies
	for (int j = 0; j < cfg->Jdes; j++) {
		// Prepare segment loop parameters
		unsigned long int Lj = round(cfg->fsamp * m / (cfg->fmin * exp(j*g/(cfg->Jdes - 1))));
		// Make sure Lj is even (tmp?)
		Lj = (Lj % 2) ? Lj - 1 : Lj;
		unsigned long int delta_segment = floor(Lj * (1.0 - (double) (cfg->ovlp / 100.)));
		unsigned int n_segments = floor(1 + (nread - Lj) / delta_segment);
		/* Adjust for edge case */
		long int tmp = (n_segments - 1)*delta_segment + Lj;
		if (tmp == nread) n_segments--;

		// Initialise segment loop vars
		double shifted_m = (double) m * (Lj - 1) / (double) Lj;
		double kernel_norm = (Lj - 1) / (double)Nfft;
		// Get position in FFT bin freq
		double search_freq = cfg->fsamp * m / (double)Lj;  // Position of spectral peak in Hz
		double fft_resolution = (double) cfg->fsamp / (double) Nfft;
		unsigned long int ikernel = round(search_freq / fft_resolution);
		double ref_kernel = get_kernel(ikernel, kernel_norm, shifted_m);
		double kernel_val = ref_kernel;

		// Get delta_i
		unsigned long int delta_i = 0;
		int max_buffered_kernel_values = 500;
		double kernel_values[max_buffered_kernel_values];
		kernel_values[0] = get_kernel((double)ikernel, kernel_norm, shifted_m);

		while (fabs(kernel_val) > ref_kernel * cfg->constQ_rel_threshold) {
			delta_i++;
			if (ikernel + delta_i >= Nfft) break;
			kernel_val = get_kernel(ikernel + delta_i, kernel_norm, shifted_m);
			if (delta_i < max_buffered_kernel_values) kernel_values[delta_i] = kernel_val;
		}

		// Read FFT information from disk //TODO: Can reduce repetitions?
		long unsigned int fft_offset;
		if (Nfft > max_samples_in_memory) {
			fft_real = (double*) xmalloc((2*delta_i + 1) * sizeof(double));
			fft_imag = (double*) xmalloc((2*delta_i + 1) * sizeof(double));
			fft_offset = ikernel - delta_i;
			hsize_t offset[2] = {0, fft_offset};
			hsize_t count[2] = {1, 2*delta_i + 1};
			hsize_t data_rank = 1;
			hsize_t data_count[1] = {count[1]};
			read_from_dataset(_contents_ptr, offset, count, data_rank, data_count, fft_real);
			offset[0] = 1;
			read_from_dataset(_contents_ptr, offset, count, data_rank, data_count, fft_imag);
		} else {
			fft_offset = 0;
		}

		// Loop over segments
		double total = 0, segment_real, segment_imag, fft_freq, sign, shift_real, shift_imag, exp_factor;
		for (int k = 0; k < n_segments; k++) {
			// Sum over +- delta_i & normalise
			segment_real = 0;
			segment_imag = 0;
			for (unsigned long int _delta_i = 0; _delta_i <= delta_i; _delta_i++) {
				// Adjust data location by multiplying exp's to the spectral terms
				unsigned long int i_fft = ikernel + _delta_i;
				exp_factor = -2*M_PI*(double)i_fft/(double)Nfft*(double)(0.5*(Nfft - Lj) - k*delta_segment);
				shift_real = cos(exp_factor);
				shift_imag = sin(exp_factor);

				if (_delta_i >= max_buffered_kernel_values)
					get_kernel(i_fft, kernel_norm, shifted_m);
				else
					kernel_val = kernel_values[_delta_i];
				sign = i_fft % 2 ? -1 : +1;

				// Complex multiplication
				segment_real += kernel_val*sign * (fft_real[i_fft - fft_offset]*shift_real - fft_imag[i_fft - fft_offset]*shift_imag);
				segment_imag += kernel_val*sign * (fft_real[i_fft - fft_offset]*shift_imag + fft_imag[i_fft - fft_offset]*shift_real);

				// Now the other side
				if (_delta_i == 0 || ikernel - _delta_i < 1) continue;
				i_fft = ikernel - _delta_i;
				exp_factor = -2*M_PI*(double)i_fft/(double)Nfft*(double)(0.5*(Nfft - Lj) - k*delta_segment);
				shift_real = cos(exp_factor);
				shift_imag = sin(exp_factor);

				// I have to re-calculate the kernel_val here because the rounding of the indices makes the kernel slightly asymmetric
				kernel_val = get_kernel(i_fft, kernel_norm, shifted_m);
				segment_real += kernel_val*sign * (fft_real[i_fft - fft_offset]*shift_real - fft_imag[i_fft - fft_offset]*shift_imag);
				segment_imag += kernel_val*sign * (fft_real[i_fft - fft_offset]*shift_imag + fft_imag[i_fft - fft_offset]*shift_real);
			}
			total += (segment_real*segment_real + segment_imag*segment_imag);
		}
		total *= pow((double) Lj / (double) Nfft, 2);

		// Clean
		if (Nfft > max_samples_in_memory) {
			// TODO: There has to be a better way
			xfree(fft_real);
			xfree(fft_imag);
			fft_real = fft_imag = NULL;
		}
		// - Fill data arrays
		double norm_psd = 2. / ((double)n_segments * cfg->fsamp * norm_propto_factor * (double)Lj);
		data->psd[j] = total * norm_psd;
		data->avg[j] = n_segments;

		// Progress tracking
		if (j % 100 == 0) {
			progress = 100. * (double) j / cfg->Jdes;
			printf ("\b\b\b\b\b\b%5.1f%%", progress);
			fflush (stdout);
        }
	}

	// Clean-up
	if (data_real) xfree(data_real);
	if (data_imag) xfree(data_imag);
	if (fft_real) xfree(fft_real);
	if (fft_imag) xfree(fft_imag);

	// Finish
    close_hdf5_contents(&contents);
    if (_contents_ptr) close_hdf5_contents(_contents_ptr);
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
  if ((*cfg).METHOD == 0) calculate_lpsd(cfg, data);
  else if ((*cfg).METHOD == 1) calculate_fft_approx(cfg, data);
  else if ((*cfg).METHOD == 2) calculate_constQ_approx(cfg, data);
  else gerror("Method not implemented.");
}
