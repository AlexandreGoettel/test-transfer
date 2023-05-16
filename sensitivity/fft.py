from tqdm import tqdm, trange
import numpy as np
import h5py


def smallest_power_of_two_above(N):
    return 2 ** (N-1).bit_length()


def fill_ordered_coefficients(n):
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

    # Now loop over the rest of the pyramid
    progress_bar = tqdm(total=np.sum(2**np.arange(n_depth)))
    while n_depth > 0:
        # Iterate n_depth
        n_depth -= 1
        two_to_n_depth = 2 ** n_depth
        Nj0_over_two_n_depth = Nj0 // two_to_n_depth
        Nfft_over_two_n_depth = round(Nfft / two_to_n_depth)
        # Number of memory units in lower-level pyramid segment
        n_mem_units = 2 ** round(np.log2(Nfft) - np.log2(Nmax) - n_depth - 1)

        # Loop over segments at this pyramid level
        for i_pyramid in range(two_to_n_depth):
            # Loop over memory units (of length Nmax) in one lower-level segment
            for j in range(n_mem_units):
                # Load even terms
                offset = (j + 2*i_pyramid*n_mem_units)*Nmax
                even_terms = _contents[offset:offset + Nmax]

                # Load odd terms
                offset += n_mem_units * Nmax
                odd_terms = _contents[offset: offset + Nmax]

                # Piecewise (complex) multiply odd terms with exp term
                isReverse = -1 if reverse else 1
                exp_factor = isReverse * -2.0j * np.pi / Nfft_over_two_n_depth
                k = np.arange(Nmax)
                complex_exp = np.exp((j * Nmax + k) * exp_factor)
                odd_terms *= complex_exp

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
        progress_bar.update(two_to_n_depth)
    progress_bar.close()

def compare_memory_FFT_and_np_fft():
    # Generate random input data
    Nj0 = 1000
    Nfft = smallest_power_of_two_above(Nj0)
    input_data = np.random.rand(Nj0).astype(np.complex128)

    # Save input data to h5py file
    with h5py.File('input_data.h5', 'w') as f:
        f.create_dataset('data', data=input_data, dtype=np.complex128)

    # Call memory_FFT function
    with h5py.File('input_data.h5', 'r') as f1, h5py.File('memory_FFT_output.h5', 'w') as f2:
        f2.create_dataset('data', (Nfft,), dtype=np.complex128)
        memory_FFT(1000, 1024, 512, f1, f2, reverse=False)

    # Get memory_FFT output from h5py file
    with h5py.File('memory_FFT_output.h5', 'r') as f:
        memory_FFT_output = np.array(f['data'])

    # Call np.fft.fft
    padded_input_data = np.zeros(Nfft)
    padded_input_data[:Nj0] = input_data
    np_fft_output = np.fft.fft(padded_input_data)

    # Compare memory_FFT output and np.fft.fft output
    idx = np.array([0, 1, Nj0//2, -1])
    for x, x_np in zip(memory_FFT_output[idx], np_fft_output[idx]):
        print(x.real, x_np.real)
        print(x.imag, x_np.imag)
        print("------")
    print("Results are equal:", np.allclose(memory_FFT_output, np_fft_output))


def compare_memory_iFFT_and_np_ifft():
    # Generate random input data
    Nj0 = 1000
    Nfft = smallest_power_of_two_above(Nj0)
    input_data = np.random.rand(Nj0).astype(np.complex128)

    # Save input data to h5py file
    with h5py.File('input_data.h5', 'w') as f:
        f.create_dataset('data', data=input_data, dtype=np.complex128)

    # Call memory_FFT function
    with h5py.File('input_data.h5', 'r') as f1, h5py.File('memory_FFT_output.h5', 'w') as f2:
        f2.create_dataset('data', (Nfft,), dtype=np.complex128)
        memory_FFT(1000, 1024, 512, f1, f2, reverse=True)

    # Get memory_FFT output from h5py file
    with h5py.File('memory_FFT_output.h5', 'r') as f:
        memory_FFT_output = np.array(f['data'])

    # Call np.fft.fft
    padded_input_data = np.zeros(Nfft)
    padded_input_data[:Nj0] = input_data
    np_fft_output = np.fft.ifft(padded_input_data)

    # Compare memory_FFT output and np.fft.fft output
    idx = np.array([0, 1, Nj0//2, -1])
    for x, x_np in zip(memory_FFT_output[idx], np_fft_output[idx]):
        print(x.real, x_np.real)
        print(x.imag, x_np.imag)
        print("------")
    print("Results are equal:", np.allclose(memory_FFT_output, np_fft_output))


def compare_fft_np_fft():
    x = np.array(np.random.normal(size=1024), dtype=np.complex128)
    y_np = np.fft.fft(x)
    y = FFT(x, reverse=False)

    yi = FFT(x, reverse=True)
    yi_np = np.fft.ifft(x)

    for x, x_np in zip(y[:3], y_np[:3]):
        print(f"{x.real:.2f}\t{x_np.real:.2f}")
        print(f"{x.imag:.2f}\t{x_np.imag:.2f}")
    print("Results are equal:", np.allclose(y, y_np))
    for x, x_np in zip(yi[:3], yi_np[:3]):
        print(f"{x.real:.2e}\t{x_np.real:.2e}")
        print(f"{x.imag:.2e}\t{x_np.imag:.2e}")
    print("Results are equal:", np.allclose(yi, yi_np))


if __name__ == '__main__':
    compare_memory_iFFT_and_np_ifft()
    compare_fft_np_fft()
    compare_memory_FFT_and_np_fft()
