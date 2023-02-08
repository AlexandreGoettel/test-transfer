#ifndef __lpsd_h
#define __lpsd_h

void calculateSpectrum(tCFG *cfg, tDATA *data);
void calculate_lpsd(tCFG*, tDATA*);
static void calc_params(tCFG*, tDATA*);
static void getDFT2(int, double, double, double, int,
                    double*, int*, struct hdf5_contents*);

void calculate_fft_approx(tCFG*, tDATA*);

#endif
