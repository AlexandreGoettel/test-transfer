#ifndef __lpsd_h
#define __lpsd_h

void do_DFT_iteration(double*, double*, hsize_t*, hsize_t*, struct hdf5_contents*, double*, double*);
double process_segment(double*, double*, int, int, struct hdf5_contents*);
void calculateSpectrum(tCFG *cfg, tDATA *data);

#endif
