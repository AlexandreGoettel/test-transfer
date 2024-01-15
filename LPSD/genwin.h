#ifndef __genwin_h
#define __genwin_h

#include <stdbool.h>

/* ANSI prototypes of externally visible functions: */

void set_window (int type, double req_psll, char *name, double *psll,
		 double *rov, double *nenbw, double *w3db, double *flatness,
		 double *sbin);

void makewinsincos (long int nfft, double bin, double *win, double *winsum,
		            double *winsum2, double *nenbw);
void makewinsincos_indexed (unsigned long int nfft, double bin, double *win, double *winsum,
		            double *winsum2, double *nenbw, unsigned long int, unsigned int, bool);

void makewin (unsigned int nfft, double *win,
              double *winsum, double *winsum2, double *nenbw);
void makewin_indexed (unsigned long int nfft, unsigned long int offset, unsigned int count,
              double *win, double *winsum, double *winsum2, double *nenbw,
              bool reset_sums);

#endif
