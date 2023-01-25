#ifndef __lpsd_h
#define __lpsd_h

void doDFTIteration(double*, double*, hsize_t*, hsize_t*, struct hdf5_contents*, double*, double*);
void calculateSpectrum(tCFG *cfg, tDATA *data);

#endif
