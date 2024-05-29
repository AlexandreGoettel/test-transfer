#ifndef FFT_H
#define FFT_H

// Custom imports
#include "IO.h"
#include "config.h"

// Function declaration
long int ld_pow(long int, long int);
int ld_log2(unsigned long int);
int count_set_bits (long int);
unsigned long int get_next_power_of_two (unsigned long int);
void fill_ordered_coefficients(int, int*);
void stride_over_array (double*, int, int, int, double*);

void FFT(double*, double*, unsigned int, double*, double*);
void FFT_control_memory(unsigned long int, unsigned long int, unsigned int,
                        unsigned long int, struct hdf5_contents*,
                        struct hdf5_contents*, struct hdf5_contents*);

#endif // FFT_H
