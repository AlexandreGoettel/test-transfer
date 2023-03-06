#ifndef __lpsd_h
#define __lpsd_h

double get_mean(int*, int);
int count_set_bits(int);
int get_next_power_of_two(int);
void stride_over_array(double*, int, int, int, double*);
int get_N_j(int, double, double, double, int);  // TODO: replace with nffts
double get_f_j(int, double, double, int);  // TODO: replace with fspec
void fill_ordered_coefficients(int, int*);

void calculateSpectrum(tCFG *cfg, tDATA *data);
void calculate_lpsd(tCFG*, tDATA*);
static void calc_params(tCFG*, tDATA*);
static void getDFT2(int, double, double, double,
                    double*, int*, struct hdf5_contents*);

void calculate_fft_approx(tCFG*, tDATA*);
void FFT(double*, double*, int, double*, double*);
void FFT_control_memory(int, int, int, int, struct hdf5_contents*,
                        struct hdf5_contents*, struct hdf5_contents*);

#endif