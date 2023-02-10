#ifndef __lpsd_h
#define __lpsd_h

void calculateSpectrum(tCFG *cfg, tDATA *data);
void calculate_lpsd(tCFG*, tDATA*);
static void calc_params(tCFG*, tDATA*);
static void getDFT2(int, double, double, double, int,
                    double*, int*, struct hdf5_contents*);

void calculate_fft_approx(tCFG*, tDATA*);
void FFT(double*, double*, int, double, double*, double*);

#endif


//def FFT(x, N):
//    """For this to work, N must be a power of two."""
//    if N == 1:
//        return x
//    m = int(N/2)
//    X_even = FFT(x[::2], m)  # N/2 sequence
//    X_odd = np.exp(-2j*np.pi*np.arange(m)/N) * FFT(x[1::2], m)  # N/2 sequence
//    #factor = np.exp(-2j*np.pi*np.arange(m)/N)  # N/2 sequence
//
//    X = np.concatenate([X_even + X_odd,
//                        X_even - X_odd])  # N sequence
//    return X
