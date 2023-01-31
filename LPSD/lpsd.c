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

/* 
	copies nfft values from data to segm
	if drift removal is selected, a linear regression of data is 
	performed and the subtracted values are copied to segm
*/
//static void
//remove_drift (double *segm, double *data, int nfft, int LR)
//{
//  int i;
//  long double sx, sy, stt, sty, xm, t;
//  double a,b;
//  if (LR == 2)
//    {				/* subtract straight line through first and last point */
//      a = data[0];
//      b = data[nfft - 1] - data[0] / (double) (nfft - 1.0);
//      for (i = 0; i < nfft; i++)
//	{
//	  segm[i] = data[i] - (a + b * i);
//	}
//    }
//  else if (LR == 1)
//    {				/* linear regression */
//
//      sx = sy = 0;
//      for (i = 0; i < nfft; i++)
//	{
//	  sx += i;
//	  sy += data[i];
//	}
//      xm = sx / nfft;
//      stt = sty = 0;
//      for (i = 0; i < nfft; i++)
//	{
//	  t = i - xm;
//	  stt += t * t;
//	  sty += t * data[i];
//	}
//      b = sty / stt;
//      a = (sy - sx * b) / nfft;
//      for (i = 0; i < nfft; i++)
//	{
//	  segm[i] = data[i] - (a + b * i);
//	}
//    }
//  else if (LR == 0)
//    {				/* copy data */
//      for (i = 0; i < nfft; i++)
//	{
//	  segm[i] = data[i];
//	}
//    }
//}

// @brief Put "count" samples in "segment" (memory), then loop over it to calculate DFT.
void do_DFT_iteration(double *segment, double *window_pointer, hsize_t *offset,
                      int offset_window, hsize_t *count, double bin, int total_samples,
                      struct hdf5_contents *contents, double *dft_re, double *dft_im)
{
    // Put strain data in memory
    read_from_dataset(contents, offset, count, segment);

    // Calculate window
    makewinsincos_indexed(total_samples, bin, window_pointer, &winsum, &winsum2,
                          &nenbw, offset_window, (int)count[0], offset_window == 0);

    // Loop over data to calculate DFT
    double sample;
    register int i;
    for (int i = 0; i < (int)count[0]; i++) {
        sample = segment[i];
        *dft_re += window_pointer[i*2] * sample;
        *dft_im += window_pointer[i*2 + 1] * sample;
    }
}


// @brief Calculate DFT over segment, only put max_samples_in_memory at a time
double process_segment(double *segment, double *window, int max_samples_in_memory,
                       int start, int remaining_samples, double bin, struct hdf5_contents *contents)
{
  int loop_index = 0;  /* Keep track of while loop iterations */
  double dft_re, dft_im;	/* Real and imaginary parts of DFT */
  dft_re = dft_im = 0;
  hsize_t offset[1], count[1];
  int offset_window;
  int total_samples = remaining_samples;  /* To normalise window */

  while (remaining_samples > 0)
  {
    if (remaining_samples < max_samples_in_memory) {
      count[0] = remaining_samples;
      remaining_samples = 0;
    } else {
      count[0] = max_samples_in_memory;
      remaining_samples -= max_samples_in_memory;
    }

    offset[0] = start + loop_index * max_samples_in_memory;
    offset_window =  loop_index * max_samples_in_memory;
    do_DFT_iteration(segment, window, offset, offset_window, count, bin,
                     total_samples, contents, &dft_re, &dft_im);
    loop_index++;
  }
  return dft_re * dft_re + dft_im * dft_im;
}


static void
getDFT2 (char* filename, char* dataset_name, int nfft, double bin, double fsamp,
         double ovlp, int LR, double *rslt, int *avg)
{
  double total;		/* Running sum of DFTs */

  /* Prepare data */
  // TODO: this opens the file for each iteration.. ok?
  struct hdf5_contents *contents = read_hdf5_file(filename, dataset_name);

  /* Configure variables for DFT */
//  int max_samples_in_memory = 5*6577770;  // Around 500 MB //TODO: this shouldn't be hard-coded!
  int max_samples_in_memory = 500;  // tmp
  if (max_samples_in_memory > nfft) max_samples_in_memory = nfft; // Don't allocate more than you need

  /* Allocate data and window memory segments */
  double *strain_data_segment = (double*) xmalloc(max_samples_in_memory * sizeof(double));
  double *window = (double*) xmalloc(2*max_samples_in_memory * sizeof(double));
  assert(window != 0 && strain_data_segment != 0);

  // Calculate DFT over first segment
  int start = 0;
  int nsum = 1;
  total = process_segment(strain_data_segment, window, max_samples_in_memory,
                          start, nfft, bin, contents);

  // Process other segments if available
  start += nfft * (1.0 - (double) (ovlp / 100.));
  while (start + nfft < nread)
    {
      /* Calculate DFT */
      total += process_segment(strain_data_segment, window, max_samples_in_memory,
                               start, nfft, bin, contents);
      nsum++;
      start += nfft * (1.0 - (double) (ovlp / 100.));	/* go to next segment */
    }

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
  close_hdf5_contents(contents);
}


/*
	calculates paramaters for DFTs
	
	input
		nread		number of data
		fsamp		sampling frequency
		ndiv		desired number of entries in spectrum
		sollavg		desired number of averages
	output
		ndiv		actual number of entries in spectrum
		fspec		frequencies in spectrum
		bins		bins for DFTs
		nffts		dimensions for DFTs
 ********************************************************************************
 	Naming convention	source code	publication
				i		j
				fresc		r_{min}
				fresb		r_{avg}
				fresa		r'
				fres		r''
				ndft		L(j)
				bin		m(j)
 ********************************************************************************/
static void
calc_params (tCFG * cfg, tDATA * data)
{
  double fres, f;
  int i, i0, ndft;
  double bin;
  double navg;
  double ovfact, xov, g;

  ovfact = 1. / (1. - (*cfg).ovlp / 100.);
  xov = (1. - (*cfg).ovlp / 100.);
  g = log ((*cfg).fmax / (*cfg).fmin);

  i = (*cfg).nspec * (*cfg).iter;
  i0 = i;
  f = (*cfg).fmin * exp (i * g / ((*cfg).Jdes - 1.));
  while (f <= (*cfg).fmax && i / (*cfg).nspec < (*cfg).iter + 1)
   {
      fres = f * (exp (g / ((*cfg).Jdes - 1.)) - 1);
      ndft = round ((*cfg).fsamp / fres);
      bin = (f / fres);
      navg = ((double) ((nread - ndft)) * ovfact) / ndft + 1;
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
  for (k = k_start; k < (*cfg).nspec; k++)
    {
      getDFT2((*cfg).ifn, (*cfg).dataset_name, (*data).nffts[k], (*data).bins[k], (*cfg).fsamp, (*cfg).ovlp,
	      (*cfg).LR, &rslt[0], &(*data).avg[k]);

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
  printf ("\b\b\b\b\b\b  100%%\n");
  fflush (stdout);
  gettimeofday (&tv, NULL);
  printf ("Duration (s)=%5.3f\n\n", tv.tv_sec - start + tv.tv_usec / 1e6);
}

//void
//calculate_fftw (tCFG * cfg, tDATA * data)
//{
//  int nfft;			/* dimension of DFT */
//  FILE *wfp;
//  fftw_plan plan;
//  double *rawdata;		/* start address of data */
//  double *out;
//  double *segm;			/* contains data of one segment without drift */
//  int i, j;
//  double d;
//  int start;
//  double *west_sumw;
//  double west_q, west_r, west_temp;
//  int navg;
//  double *fft_ps, *fft_varps;
//
//  struct timeval tv;
//  double stt;
//
//  gettimeofday (&tv, NULL);
//  stt = tv.tv_sec + tv.tv_usec / 1e6;
//
//  nfft = (*cfg).nfft;
//
//  dwin = (double *) xmalloc (nfft * sizeof (double));
//  segm = (double *) xmalloc (nfft * sizeof (double));
//  out = (double *) xmalloc (nfft * sizeof (double));
//  west_sumw = (double *) xmalloc ((nfft / 2 + 1) * sizeof (double));
//  fft_ps = (double *) xmalloc ((nfft / 2 + 1) * sizeof (double));
//  fft_varps = (double *) xmalloc ((nfft / 2 + 1) * sizeof (double));
//
//  /* calculate window function */
//  makewin (nfft, 0, dwin, &winsum, &winsum2, &nenbw);
//
//  /* import fftw "wisdom" */
//  if ((wfp = fopen ((*cfg).wfn, "r")) == NULL)
//    message1 ("Cannot open '%s'", (*cfg).wfn);
//  else
//    {
//      if (fftw_import_wisdom_from_file (wfp) == 0)
//	message ("Error importing wisdom");
//      fclose (wfp);
//    }
//  /* plan DFT */
//  printf ("Planning...");
//  fflush (stdout);
//
//  plan = fftw_plan_r2r_1d (nfft, segm, out, FFTW_R2HC, FFTW_ESTIMATE);
//  printf ("done.\n");
//  fflush (stdout);
//
//  rawdata = get_data ();
//  assert (rawdata != 0);
//
//  printf ("Computing output\n");
//
//  /* remove drift from first data segment */
//  remove_drift (&segm[0], &rawdata[0], nfft, (*cfg).LR);
//  /* multiply data with window function */
//  for (i = 0; i < nfft; i++)
//    segm[i] = segm[i] * dwin[i];
//
//  fftw_execute (plan);
//
//  d = 2 * (out[0] * out[0]);
//  (*data).fft_ps[0] = d;
//  west_sumw[0] = 1.;
//  (*data).fft_varps[0] = 0;
//  for (j = 1; j < nfft / 2 + 1; j++)
//    {
//      d = 2 * (out[j] * out[j] + out[nfft - j] * out[nfft - j]);
//      (*data).fft_ps[j] = d;
//      west_sumw[j] = 1.;
//      (*data).fft_varps[j] = 0;
//    }
//  navg = 1;
//  start = nfft * (1.0 - (double) ((*cfg).ovlp / 100.));
//
//  /* remaining segments */
//  while (start + nfft < nread)
//    {
//
//      printf (".");
//      fflush (stdout);
//      if (navg % 75 == 0)
//	printf ("\n");
//
//      navg++;
//      remove_drift (&segm[0], &rawdata[start], nfft, (*cfg).LR);
//
//      /* multiply data with window function */
//      for (i = 0; i < nfft; i++)
//	    segm[i] = segm[i] * dwin[i];
//
//      fftw_execute (plan);
//
//      d = 2 * (out[0] * out[0]);
//      west_q = d - (*data).fft_ps[0];
//      west_temp = west_sumw[0] + 1;
//      west_r = west_q / west_temp;
//      (*data).fft_ps[0] += west_r;
//      (*data).fft_varps[0] += west_r * west_sumw[0] * west_q;
//      west_sumw[0] = west_temp;
//
//      for (j = 1; j < nfft / 2 + 1; j++)
//	{
//	  d = 2 * (out[j] * out[j] + out[nfft - j] * out[nfft - j]);
//	  west_q = d - (*data).fft_ps[j];
//	  west_temp = west_sumw[j] + 1;
//	  west_r = west_q / west_temp;
//	  (*data).fft_ps[j] += west_r;
//	  (*data).fft_varps[j] += west_r * west_sumw[j] * west_q;
//	  west_sumw[j] = west_temp;
//	}
//      start += nfft * (1.0 - (double) ((*cfg).ovlp / 100.));	/* go to next segment */
//    }
//
//  if (navg > 1)
//    {
//      for (i = 0; i < nfft / 2 + 1; i++)
//	{
//	  (*data).fft_varps[i] =
//	    sqrt ((*data).fft_varps[i] / ((double) navg - 1));
//	}
//    }
//  else
//    {
//      for (i = 0; i < nfft / 2 + 1; i++)
//	{
//	  (*data).fft_varps[i] = (*data).fft_ps[i];
//	}
//    }
//  /* normalizations and additional information */
//  j = 0;
//  for (i = 0; i < nfft / 2 + 1; i++)
//    {
//      if (((*cfg).fres * i >= (*cfg).fmin) &&
//	  ((*cfg).fres * i <= (*cfg).fmax) && ((*cfg).sbin <= i))
//	{
//	  (*data).fspec[j] = (*cfg).fres * i;
//	  (*data).ps[j] = (*data).fft_ps[i] / (winsum * winsum);
//	  (*data).varps[j] = (*data).fft_varps[i] / (winsum * winsum);
//	  (*data).psd[j] = (*data).fft_ps[i] / ((*cfg).fsamp * winsum2);
//	  (*data).varpsd[j] = (*data).fft_varps[i] / ((*cfg).fsamp * winsum2);
//	  (*data).avg[j] = navg;
//	  (*data).nffts[j] = nfft;
//	  (*data).bins[j] = (double) i;
//	  j++;
//	}
//    }
//  printf ("done.\n");
//
//  gettimeofday (&tv, NULL);
//  printf ("Duration (s)=%5.3f\n\n", tv.tv_sec - stt + tv.tv_usec / 1e6);
//
//  /* write wisdom to file */
//  if ((wfp = fopen ((*cfg).wfn, "w")) == NULL)
//    message1 ("Cannot open '%s'", (*cfg).wfn);
//  else
//    {
//      fftw_export_wisdom_to_file (wfp);
//      fclose (wfp);
//    }
//  /* clean up */
//  fftw_destroy_plan (plan);
//
//  /* forget wisdom, free memory */
//  fftw_forget_wisdom ();
//  xfree (fft_ps);
//  xfree (fft_varps);
//  xfree (west_sumw);
//  xfree (dwin);
//  xfree (segm);
//  xfree (out);
//}

/*
	works on cfg, data structures of the calling program
*/
void
calculateSpectrum (tCFG * cfg, tDATA * data)
{
  nread = floor (((*cfg).tmax - (*cfg).tmin) * (*cfg).fsamp + 1);

  if ((*cfg).METHOD == 0)
    {
      calc_params (cfg, data);
      calculate_lpsd (cfg, data);
    }
  else if ((*cfg).METHOD == 1)
    {
      // calculate_fftw (cfg, data);
      gerror("Method 1 (fftw) is not implemented in this version.");
    }
}
