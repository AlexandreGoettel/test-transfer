from tqdm import tqdm, trange
import numpy as np


def smallest_power_of_two_above(N):
    return 2 ** (N-1).bit_length()


def fill_ordered_coefficients(n):
    """Create array with bit-reversal FFT ordering, before I realised that was a thing."""
    coefficients = np.zeros(2**n, dtype=int)
    coefficients[:2] = [0, 1]

    if n == 1:
        return coefficients

    for m in range(2, n+1):
        two_to_m_minus_one = 2 ** (m-1)
        tmp = np.zeros(2**m, dtype=int)

        # Calculate ordered coefficients using previous coefficients
        for i in range(two_to_m_minus_one):
            tmp[2*i], tmp[2*i+1] = coefficients[i], coefficients[i] + two_to_m_minus_one

        # Update coefficients
        coefficients[:2**m] = tmp

    return coefficients


def FFT(data, reverse=True, is_top_level=True):
    """Calculate the FFT using the Tukey-Cooley algorithm"""
    N = len(data)
    if N == 1:
        return data
    m = N // 2

    # Calculate FFT over halved arrays
    X_even = FFT(data[::2], reverse, False)
    X_odd = FFT(data[1::2], reverse, False)

    # Calculate exponential term to multiply to X_odd
    isReverse = -1 if reverse else 1
    X_odd *= np.exp(isReverse * -2.j * np.pi * np.arange(m) / N)

    # Calculate final answer
    output = np.zeros(N, dtype="complex128")
    output[:m] = X_even + X_odd
    output[m:] = X_even - X_odd

    if reverse and is_top_level:
        output /= N
    return output


def memory_FFT(Nj0, Nfft, Nmax, fcontents, f_contents,
               dset_name="data", _dset_name="data",
               segment_offset=0, reverse=True):
    """
    Perform an FFT(or iFFT) that only loads a few multiples of Nmax in memory.
    
    contents contains the data of length Nj0
    _contents contains the output of length Nfft
    Nmax is the number used to build the pyramid
    Nmax and Nfft MUST be powers of two (and int)
    """
    n_depth = int(np.round(np.log2(Nfft) - np.log2(Nmax)))
    two_to_n_depth = 2**n_depth
    Nj0_over_two_n_depth = Nj0 // two_to_n_depth
    ordered_coefficients = fill_ordered_coefficients(n_depth)

    contents, _contents = fcontents[dset_name], f_contents[_dset_name]

    # Perform FFTs on bottom layer of pyramid and save results to temporary file
    print("Bottom layer FFT operations..")
    for i in trange(two_to_n_depth):
        # Read data
        offset = ordered_coefficients[i] + segment_offset
        Ndata = Nj0_over_two_n_depth + 1 if ordered_coefficients[i] + Nj0_over_two_n_depth*two_to_n_depth < Nj0 else Nj0_over_two_n_depth
        count = Ndata
        stride = two_to_n_depth
        data_subset = np.zeros(Nmax, dtype=np.complex128)  # Nmax in memory
        data_subset[:Ndata] = contents[offset:offset+count*stride:stride]

        # Take fft
        fft_output = FFT(data_subset, reverse, False)  # 2*Nmax in memory

        # Save to file
        offset = i*Nmax
        _contents[offset:offset+Nmax] = fft_output

    del data_subset, fft_output
    # Now loop over the rest of the pyramid
    print("Pyramid climb..")
    progress_bar = tqdm(total=2**(np.log2(Nfft) - np.log2(Nmax) - 1)*n_depth)
    while n_depth > 0:
        # Iterate n_depth
        n_depth -= 1
        two_to_n_depth = 2 ** n_depth
        Nfft_over_two_n_depth = round(Nfft / two_to_n_depth)
        # Number of memory units in lower-level pyramid segment
        n_mem_units = 2 ** round(np.log2(Nfft) - np.log2(Nmax) - n_depth - 1)

        # Loop over segments at this pyramid level
        for i_pyramid in range(two_to_n_depth):
            # Loop over memory units (of length Nmax) in one lower-level segment
            for j in range(n_mem_units):
                # Load odd terms
                offset = (j + (2*i_pyramid + 1)*n_mem_units)*Nmax
                odd_terms = _contents[offset:offset + Nmax]  # Nmax in memory

                # Piecewise (complex) multiply odd terms with exp term
                isReverse = -1 if reverse else 1
                exp_factor = isReverse * -2.0j * np.pi / Nfft_over_two_n_depth
                # Use ufunc to reduce memory usage from 3*Nmax to 2*Nmax
                # in-operation, back at Nmax after operation
                np.multiply(odd_terms,  # 2*Nmax
                            np.exp((j * Nmax + np.arange(Nmax)) * exp_factor),
                            out=odd_terms)

                # Load even terms
                offset = (j + 2*i_pyramid*n_mem_units)*Nmax
                even_terms = _contents[offset:offset + Nmax]  # 2*Nmax in memory

                # Normalise inverse FFT terms correctly
                if reverse and not n_depth:
                    odd_terms /= Nfft
                    even_terms /= Nfft

                # Combine left side
                offset_left = (j + 2 * i_pyramid * n_mem_units) * Nmax
                _contents[offset_left:offset_left + Nmax] = even_terms + odd_terms

                # Combine right side
                offset_right = (j + (2 * i_pyramid + 1) * n_mem_units) * Nmax
                _contents[offset_right:offset_right + Nmax] = even_terms - odd_terms

                # Clean up
                del even_terms  # no need for odd_terms, re-created first
                progress_bar.update(1)
    progress_bar.close()
