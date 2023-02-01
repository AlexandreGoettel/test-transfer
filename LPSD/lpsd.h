#ifndef __lpsd_h
#define __lpsd_h

void process_memory_unit(double*, double*, int, int, int, int, double,
                         struct hdf5_contents*, double, double*);
void calculateSpectrum(tCFG *cfg, tDATA *data);

#endif
