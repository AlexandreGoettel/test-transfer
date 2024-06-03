#ifndef __IO_h
#define __IO_h

// Custom imports
#include "config.h"
#include "hdf5.h"

// Declarations
int exists(char *fn);
void probe_file(unsigned int t, unsigned int A, unsigned int B);
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
int hdf5_file_exists(const char*);
void read_hdf5_file(struct hdf5_contents *contents, char*, char*);
void open_hdf5_file(struct hdf5_contents *contents, char*, char*, hsize_t, hsize_t*);
void write_to_hdf5(struct hdf5_contents*, double*, hsize_t*, hsize_t*, hsize_t, hsize_t*);
void read_from_dataset(struct hdf5_contents*, hsize_t*, hsize_t*, hsize_t, hsize_t*, double*);
void read_from_dataset_stride(struct hdf5_contents*, hsize_t*, hsize_t*, hsize_t*, hsize_t, hsize_t*, double*);
void close_hdf5_contents(struct hdf5_contents*);

#endif
