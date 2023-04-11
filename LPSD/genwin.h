#ifndef __genwin_h
#define __genwin_h

#include <stdbool.h>

/* ANSI prototypes of externally visible functions: */

void set_window (int type, double req_psll, char *name, double *psll,
		 double *rov, double *nenbw, double *w3db, double *flatness,
		 double *sbin);

void makewinsincos (long int nfft, double bin, double *win, double *winsum,
		            double *winsum2, double *nenbw);
void makewinsincos_indexed (long int nfft, double bin, double *win, double *winsum,
		            double *winsum2, double *nenbw, int, int, bool);

void makewin (long int nfft, double *win,
              double *winsum, double *winsum2, double *nenbw);
void makewin_indexed (long int nfft, int offset, int count, double *win,
              double *winsum, double *winsum2, double *nenbw,
              bool reset_sums);

#endif
