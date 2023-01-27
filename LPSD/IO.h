#ifndef __IO_h
#define __IO_h

// Custom imports
#include "hdf5.h"

// Declarations
int exists(char *fn);
int getNoC(char *fn, int *comma);
void probe_file(unsigned int t, unsigned int A, unsigned int B);
void read_file(char *ifn, double ulsb, int start, int nread, int comma);
void close_file();
double *get_data();
void saveResult(tCFG * cfg, tDATA * data, tGNUTERM * gt, tWinInfo *wi, int argc, char *argv[]);
int write_gnufile(char *gfn, char *ofn, char *vfn, char *ifn, char *s, 
			double fmin, double fmax, double dmin, double dmax,
			char *id, int c);

// HDF5 files related I/O
struct hdf5_contents {
    hid_t file, dataset, dataspace;
    hsize_t rank;
    hsize_t *dims;
};
struct hdf5_contents* read_hdf5_file(char*, char*);
void read_from_dataset(struct hdf5_contents*, hsize_t*, hsize_t*, double*);
void close_hdf5_contents(struct hdf5_contents*);

#endif
