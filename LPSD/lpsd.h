#ifndef __lpsd_h
#define __lpsd_h

void do_DFT_iteration(double*, double*, hsize_t*, int, hsize_t*, double, int, struct hdf5_contents*, double*, double*);
double process_segment(double*, double*, int, int, int, double, struct hdf5_contents*);
void calculateSpectrum(tCFG *cfg, tDATA *data);

#endif
