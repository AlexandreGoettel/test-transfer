#ifndef __lpsd_h
#define __lpsd_h

// Helper functions
double interpolate(double, double, double, double, double);
double get_mean(int*, int);
long int ld_pow(long int, long int);
int ld_log2(unsigned long int);
int count_set_bits(long int);
unsigned long int get_next_power_of_two(unsigned long int);
void stride_over_array(double*, int, int, int, double*);

// LPSD calculations
long int get_N_j(unsigned int, double, double, double, unsigned int);  // TODO: replace with nffts
double get_f_j(unsigned int, double, double, unsigned int);  // TODO: replace with fspec
void fill_ordered_coefficients(int, int*);

// Main functions
void calculateSpectrum(tCFG *cfg, tDATA *data);
void calculate_lpsd(tCFG*, tDATA*);
static void calc_params(tCFG*, tDATA*);
static void getDFT2(long int, double, double, double,
                    double*, int*, struct hdf5_contents*);

// Block approximation
void calculate_fft_approx(tCFG*, tDATA*);
void FFT(double*, double*, int, double*, double*);
void FFT_control_memory(unsigned long int, unsigned long int, unsigned int,
                        unsigned long int, struct hdf5_contents*,
                        struct hdf5_contents*, struct hdf5_contents*);

#endif
