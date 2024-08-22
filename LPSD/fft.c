#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "fft.h"
#include "misc.h"

// @brief implementation of pow() that works with long ints as base and exponent
long int ld_pow(long int base, long int exponent) {
    long int result = 1;
    while (exponent > 0) {
        result *= base;
        exponent--;
    }
    return result;
}

// @brief implementation of int(log2()) that works with long ints
int ld_log2(unsigned long int n) {
    int count = -1;
    while (n > 0) {
        n >>= 1;
        count++;
    }
    return count;
}

// @brief recursive function to count set bits
int
count_set_bits (long int n)
{
    if (n == 0)  // base case
        return 0;
    else  // if last bit set add 1 else add 0
        return (n & 1) + count_set_bits(n >> 1);
}

// @brief Get next power of two using bit operations
unsigned long int
get_next_power_of_two (unsigned long int n)
{
    unsigned long int output;
    if (!(count_set_bits(n) == 1 || n == 0))
      output = (unsigned long int) ld_pow(2, (long int) (ld_log2(n) + 1));
    return output;
}

// @brief Stride over array up to length N, write to double *output
void
stride_over_array (double *data, int N, int stride, int offset, double *output)
{
    for (int i = offset; i < N; i += stride) *(output++) = data[i];
}

// @brief Fill coefficients with permutation order for mem-FFT access
void
fill_ordered_coefficients(int n, int *coefficients) {
    coefficients[0] = 0;
    coefficients[1] = 1;
    if (n == 1) return;
    for (int m = 2; m <= n; m++) {
        int two_to_m = pow(2, m);
        int two_to_m_minus_one = pow(2, m-1);
        int tmp[two_to_m];
        for (int i = 0; i < two_to_m_minus_one; i++) {
            tmp[2*i] = coefficients[i];
            tmp[2*i+1] = coefficients[i] + two_to_m_minus_one;
        }
        for (int i = 0; i < two_to_m; i++) coefficients[i] = tmp[i];
    }
}

// @brief Calculate FFT on data of length N
// @brief This implementation puts everything in memory, serves as test
// @brief Takes in real data
// @param Custom bin number, necessary for logarithmic frequency spacing
// @brief Memory contents reach 3N in main loop (+ 2N from recursion)
// TODO: implement Bergland's algorithm
// TODO: sin/cos optimisation
void
FFT(double *data_real, double *data_imag, unsigned int N,
    double *output_real, double *output_imag)
{
    if (N == 1) {
        output_real[0] = data_real[0];
        output_imag[0] = data_imag[0];
        return;
    }
    unsigned int m = N / 2;

    // Separate even part in real/imaginary
    double *x_even_real = (double*) xmalloc(m*sizeof(double));
    double *x_even_imag = (double*) xmalloc(m*sizeof(double));
    stride_over_array(data_imag, N, 2, 0, x_even_imag);
    stride_over_array(data_real, N, 2, 0, x_even_real);
    // Calculate FFT over halved arrays
    double *X_even_real = (double*) xmalloc(m*sizeof(double));
    double *X_even_imag = (double*) xmalloc(m*sizeof(double));
    FFT(x_even_real, x_even_imag, m, X_even_real, X_even_imag);
    // Clean up
    xfree(x_even_real);
    xfree(x_even_imag);

    // Repeat for odd part
    double *x_odd_real = (double*) xmalloc(m*sizeof(double));
    double *x_odd_imag = (double*) xmalloc(m*sizeof(double));
    stride_over_array(data_real, N, 2, 1, x_odd_real);
    stride_over_array(data_imag, N, 2, 1, x_odd_imag);
    // Calculate FFT over halved arrays
    double *X_odd_real = (double*) xmalloc(m*sizeof(double));
    double *X_odd_imag = (double*) xmalloc(m*sizeof(double));
    FFT(x_odd_real, x_odd_imag, m, X_odd_real, X_odd_imag);
    // Clean up
    xfree(x_odd_real);
    xfree(x_odd_imag);

    // Calculate exponential term to multiply to X_odd
    double exp_factor = 2.0 * M_PI / ((double) N);
    for (int i = 0; i < m; i++) {
        double y = cos(i*exp_factor);
        double x = -sin(i*exp_factor);
        double b = X_odd_real[i];
        double a = X_odd_imag[i];
        X_odd_real[i] = b*y - a*x;
        X_odd_imag[i] = a*y + b*x;
    }

    // Calculate final answer
    for (int i = 0; i < m; i++) {
        output_real[i] = X_even_real[i] + X_odd_real[i];
        output_imag[i] = X_even_imag[i] + X_odd_imag[i];
        output_real[i+m] = X_even_real[i] - X_odd_real[i];
        output_imag[i+m] = X_even_imag[i] - X_odd_imag[i];
    }

    // Clean up
    xfree(X_even_real);
    xfree(X_odd_real);
    xfree(X_even_imag);
    xfree(X_odd_imag);
}


// Perform an FFT while controlling how much gets in memory by manually calculating the
// top layers of the pyramid over sums
void
FFT_control_memory(unsigned long int Nj0, unsigned long int Nfft, unsigned int Nmax,
                   unsigned long int segment_offset, struct hdf5_contents *contents,
                   struct hdf5_contents *window_contents, struct hdf5_contents *_contents)
{
    // Determine manual recursion depth
    // Nfft and Nmax must be powers of two!!
    int n_depth = ld_log2(Nfft) - ld_log2(Nmax);

    // Get 2^n_depth data samples, then iteratively work down to n = 1
    unsigned int two_to_n_depth = (unsigned int) ld_pow(2, n_depth);  // use long for compatibility with ld_pow, but this number should be small enough
    unsigned int Nj0_over_two_n_depth = Nj0 / two_to_n_depth;  // equivalent to floor(..)
    int ordered_coefficients[two_to_n_depth];
    fill_ordered_coefficients(n_depth, ordered_coefficients);

    // Approx (5 * 16 * Nmax) bits in memory
    double *data_subset_real = (double*)malloc(Nmax*sizeof(double));
    double *data_subset_imag = (double*)malloc(Nmax*sizeof(double));
    memset(data_subset_imag, 0, Nmax*sizeof(double));
    double *fft_output_real = (double*)malloc(Nmax*sizeof(double));
    double *fft_output_imag = (double*)malloc(Nmax*sizeof(double));
    // TODO no need if no window
    double *window_subset = (double*)malloc((Nj0_over_two_n_depth+1)*sizeof(double));

    // Perform FFTs on bottom layer of pyramid and save results to temporary file
    for (unsigned int i = 0; i < two_to_n_depth; i++) {
        // Read data
        hsize_t offset[1] = {ordered_coefficients[i] + segment_offset};
        unsigned int Ndata = ordered_coefficients[i] + (unsigned long int)Nj0_over_two_n_depth*two_to_n_depth < Nj0 ?
            Nj0_over_two_n_depth + 1 : Nj0_over_two_n_depth;
        hsize_t count[1] = {Ndata};  // Note. Ndata will always be compatible with int by design
        hsize_t stride[1] = {two_to_n_depth};
        hsize_t rank = 1;
        read_from_dataset_stride(contents, offset, count, stride, rank, count, data_subset_real);
        // Zero-pad data
        for (unsigned int j = Ndata; j < Nmax; j++) data_subset_real[j] = 0;

        // Read window & apply to data (piecewise multiply)
        hsize_t window_offset[1] = {ordered_coefficients[i]};
        if (window_contents) {
			read_from_dataset_stride(window_contents, window_offset, count, stride, rank, count, window_subset);
			for (unsigned int j = 0; j < Ndata; j++) data_subset_real[j] *= window_subset[j];
		}

        // Take FFT
        FFT(data_subset_real, data_subset_imag, Nmax, fft_output_real, fft_output_imag);

        // Save real part to file
        hsize_t _offset[2] = {0, i*(unsigned long int)Nmax};
        hsize_t _count[2] = {1, Nmax};
        hsize_t _data_rank = 1;
        hsize_t _data_count[1] = {Nmax};
        write_to_hdf5(_contents, fft_output_real, _offset, _count, _data_rank, _data_count);
        // Save imaginary part
        _offset[0] = 1;
        write_to_hdf5(_contents, fft_output_imag, _offset, _count, _data_rank, _data_count);
        // Note: if I saved real/imag in one 2D array instead of two arrays,
        // I would only need one call to write_to_hdf5 in this loop. Could save time?
    }
    // Clean-up
    free(data_subset_real);
    free(data_subset_imag);
    free(window_subset);
    free(fft_output_real);
    free(fft_output_imag);

    // TODO: don't need to write the last iteration of the pyramid to file as I could work with it here directly, small speed-up
    // Put 5 * 16 * Nmax bits in memory
    double *even_terms_real = (double*)malloc(Nmax*sizeof(double));
    double *even_terms_imag = (double*)malloc(Nmax*sizeof(double));
    double *odd_terms_real = (double*)malloc(Nmax*sizeof(double));
    double *odd_terms_imag = (double*)malloc(Nmax*sizeof(double));
    double *write_vector = (double*)malloc(Nmax*sizeof(double));
    // Now loop over the rest of the pyramid
    while (n_depth > 0) {
        // Iterate n_depth
        n_depth--;
        two_to_n_depth = (unsigned int) ld_pow(2, n_depth);
        Nj0_over_two_n_depth = Nj0 / two_to_n_depth;
        unsigned long int Nfft_over_two_n_depth = round(Nfft / two_to_n_depth);
        // Number of memory units in lower-level pyramid segment
        int n_mem_units = pow(2, (int)round(ld_log2(Nfft) - ld_log2(Nmax) - n_depth - 1));

        // Loop over segments at this pyramid level
        for (unsigned int i_pyramid = 0; i_pyramid < two_to_n_depth; i_pyramid++) {
            // Loop over memory units (of length Nmax) in one lower-level segment
            for (unsigned int j = 0; j < n_mem_units; j++) {
                // Load even terms
                hsize_t offset[2] = {0, (j + 2*i_pyramid*n_mem_units)*(unsigned long int)Nmax};
                hsize_t count[2] = {1, Nmax};
                hsize_t data_rank = 1;
                hsize_t data_count[1] = {Nmax};
                read_from_dataset(_contents, offset, count, data_rank, data_count, even_terms_real);
                offset[0] = 1;
                read_from_dataset(_contents, offset, count, data_rank, data_count, even_terms_imag);

                // Load odd terms
                offset[1] += n_mem_units*(unsigned long int)Nmax;
                read_from_dataset(_contents, offset, count, data_rank, data_count, odd_terms_imag);
                offset[0] = 0;
                read_from_dataset(_contents, offset, count, data_rank, data_count, odd_terms_real);

                // Piecewise (complex) multiply odd terms with exp term
                double exp_factor = 2.0 * M_PI / ((double) Nfft_over_two_n_depth);
                for (unsigned int k = 0; k < Nmax; k++) {
                    double y = cos((j*(double)Nmax+k)*exp_factor);
                    double x = -sin((j*(double)Nmax+k)*exp_factor);
                    double a = odd_terms_imag[k];
                    double b = odd_terms_real[k];
                    odd_terms_real[k] = b*y - a*x;
                    odd_terms_imag[k] = a*y + b*x;
                }

                // Combine left side
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_real[k] + odd_terms_real[k];
                hsize_t offset_left[2] = {0, (j + 2*i_pyramid*n_mem_units)*(unsigned long int)Nmax};
                write_to_hdf5(_contents, write_vector, offset_left, count, data_rank, data_count);
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_imag[k] + odd_terms_imag[k];
                offset_left[0] = 1;
                write_to_hdf5(_contents, write_vector, offset_left, count, data_rank, data_count);

                // Combine right side
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_real[k] - odd_terms_real[k];
                hsize_t offset_right[2] = {0, (j + (2*i_pyramid + 1)*n_mem_units)*(unsigned long int)Nmax};
                write_to_hdf5(_contents, write_vector, offset_right, count, data_rank, data_count);
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_imag[k] - odd_terms_imag[k];
                offset_right[0] = 1;
                write_to_hdf5(_contents, write_vector, offset_right, count, data_rank, data_count);
            }
        }
    }
    // Clean up
    free(even_terms_real);
    free(even_terms_imag);
    free(odd_terms_real);
    free(odd_terms_imag);
    free(write_vector);
}
