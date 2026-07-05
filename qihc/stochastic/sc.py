import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Global cache dictionary
lookup_cache = {}

# Hardware Characteristics
GATE = {'energy':1, 'area':1, 'delay':1, 'power':1}
FA = {'energy':5, 'area':5, 'delay':3, 'power':5}

def load_lookup_table(mode='AND', p1=0.125, p2=0.125, scale_input=1.0, scale_mux=0.5, bit_length=8):

    assert mode in ['AND', 'OR', 'XOR', 'NOT', 'MUX', 'XNOR'], "Invalid mode selected."
    assert 0 <= p1 <= 1, "p1 must be in [0, 1]"
    assert 0 <= p2 <= 1, "p2 must be in [0, 1]"
    assert scale_input in np.arange(0.1, 1.1, 0.1), "scale_input must be in [0.1, 1] with step 0.1"
    assert scale_mux in np.arange(0, 1.1, 0.1), "scale_mux must be in [0, 1] with step 0.1"
    assert bit_length in [1, 2, 4, 8, 16, 32, 64, 128, 256], "bit_length must be one of [8, 16, 32, 64, 128, 256]"

    # Check if data is already cached
    if mode not in lookup_cache:
        print(f"Loading lookup table for mode: {mode}...")
        # Assume different modes load different files
        lookup_cache[mode] = pd.read_parquet(DATA_DIR / f"{mode}.parquet").to_numpy()
    # else:
    #     print(f"Lookup table for mode '{mode}' already loaded.")

    logic = lookup_cache[mode]

    # Quantification of ps
    # print("Quantifying inputs... ", 'p1:', p1, 'p2:', p2)
    p1 = np.round(p1 * bit_length) / bit_length
    p2 = np.round(p2 * bit_length) / bit_length
    # print("Quantified inputs: ", 'p1:', p1, 'p2:', p2)

    # Check matching rows
    if mode == 'NOT':
        matching_rows = logic[
            (logic[:, 0] == bit_length) &
            (logic[:, 1] == p1) &
            (logic[:, 2] == scale_input)
        ]
    elif mode == 'MUX':
        matching_rows = logic[
            (logic[:, 0] == bit_length) &
            (logic[:, 1] == p1) &
            (logic[:, 2] == p2) &
            (logic[:, 3] == scale_input) &
            (logic[:, 4] == scale_mux)
        ]
    else:  # AND, OR, XOR, XNOR
        matching_rows = logic[
            (logic[:, 0] == bit_length) &
            (logic[:, 1] == p1) &
            (logic[:, 2] == p2) &
            (logic[:, 3] == scale_input)
        ]

    if matching_rows.size > 0:
        mu_hat = matching_rows[0, -2]
        sigma_hat = matching_rows[0, -1]
        error = np.random.normal(mu_hat, sigma_hat, 1)
        if mode == 'NOT':
            simulated_sum = 1 - scale_input * p1 + error
        elif mode == 'MUX':
            simulated_sum = scale_input * (p1 * scale_mux + p2 * (1 - scale_mux)) + error
        else:  # AND, OR, XOR, XNOR
            simulated_sum = {
                'AND': p1 * p2,
                'OR': p1 + p2 - p1 * p2,
                'XOR': p1 + p2 - 2 * p1 * p2,
                'XNOR': 1 - (p1 + p2 - 2 * p1 * p2)
            }[mode] * scale_input + error

        # quantification
        simulated_sum = simulated_sum/scale_input
        simulated_sum = np.round(simulated_sum * bit_length) / bit_length
        # ensure within [0, 1]
        return min(max(simulated_sum, 0), 1)
    elif mode == 'MUX' and scale_mux in [0.0, 1.0]:
        simulated_sum = scale_input * (p1 if scale_mux == 1.0 else p2)
        return min(max(simulated_sum, 0), 1)
    else:
        print("No matching row found in the lookup table. Please check the input parameters.", 'mode:', mode, 'p1:', p1, 'p2:', p2, 'scale_input:', scale_input, 'scale_mux:', scale_mux, 'bit_length:', bit_length)
        return None

def PPA_stochastic_add(n_bits):
    seq_len = 2**n_bits
    return {
        'energy': seq_len * GATE['energy'],
        'area': GATE['area'],
        'delay': seq_len * GATE['delay'],
        'power': GATE['power'] * 0.1  # Accounting for low activity
    }

def PPA_binary_add(n_bits):
    return {
        'energy': n_bits * FA['energy'],
        'area': n_bits * FA['area'],
        'delay': n_bits * FA['delay'],
        'power': n_bits * FA['power'] * 0.3
    }

def PPA_stochastic_mult(n_bits):
    seq_len = 2**n_bits
    return {
        'energy': seq_len * GATE['energy'],
        'area': GATE['area'],
        'delay': seq_len * GATE['delay'],
        'power': GATE['power'] * 0.1
    }

def PPA_binary_mult(n_bits):
    # Wallace tree multiplier approximation > scales to 1.5 power. here we put naive implementation power of 2
    mult_energy = 0.7 * n_bits**2 * (GATE['energy'] + FA['energy'])
    mult_area = 0.7 * n_bits**2 * (GATE['area'] + FA['area'])
    mult_delay = 2 * np.log2(n_bits) * FA['delay']
    return {
        'energy': mult_energy,
        'area': mult_area,
        'delay': mult_delay,
        'power': 0.3 * n_bits**2 * (GATE['power'] + FA['power'])
    }

def sc_scaled_adder(p1, p2, scale_input=1.0, scale_factor=0.5, bit_length=256, PPA_mode=False):
    '''
    Addition using scaled adder SC MUX
    
    :param p1: probability input 1
    :param p2: probability input 2
    :param scale_input: scale of input probabilities
    :param scale_factor: scale factor for MUX (0.5 for average)
    :param bit_length: bit length of the stochastic representation
    :param PPA_mode: metric calculation mode

    :return: output probability after addition
    if PPA_mode is True, also return PPA score ratio (larger value is better)
    '''
    # :output (p1+p2)*scale_factor*scale_input
    if PPA_mode:
        metric = PPA_stochastic_add(bit_length)
        PPA_result = (metric['energy'] * metric['area'] * metric['delay']) ** (1/3)
        metric_binary = PPA_binary_add(bit_length)
        PPA_result_binary = (metric_binary['energy'] * metric_binary['area'] * metric_binary['delay']) ** (1/3)
        return load_lookup_table(mode='MUX', p1=p1, p2=p2, scale_input=scale_input, scale_mux=scale_factor, bit_length=bit_length), PPA_result/PPA_result_binary 
    else:
        return load_lookup_table(mode='MUX', p1=p1, p2=p2, scale_input=scale_input, scale_mux=scale_factor, bit_length=bit_length)
    
def sc_approx_adder(p1, p2, scale_input=1.0, bit_length=256, PPA_mode=False):
    '''
    Addition using approximate SC OR gate
    
    :param p1: probability input 1
    :param p2: probability input 2
    :param scale_input: scale of input probabilities
    :param bit_length: bit length of the stochastic representation
    :param PPA_mode: metric calculation mode

    :return: output probability after addition
    if PPA_mode is True, also return PPA score ratio (larger value is better)
    '''
    # :output (p1+p2-p1*p2)*scale_input (suitable for small probabilities)
    if PPA_mode:
        metric = PPA_stochastic_add(bit_length)
        PPA_result = (metric['energy'] * metric['area'] * metric['delay']) ** (1/3)
        metric_binary = PPA_binary_add(bit_length)
        PPA_result_binary = (metric_binary['energy'] * metric_binary['area'] * metric_binary['delay']) ** (1/3)
        return load_lookup_table(mode='OR', p1=p1, p2=p2, scale_input=scale_input, bit_length=bit_length), PPA_result/PPA_result_binary 
    else:
        return load_lookup_table(mode='OR', p1=p1, p2=p2, scale_input=scale_input, bit_length=bit_length)
    

def sc_multipler(p1, p2, scale_input=1.0, bit_length=256, PPA_mode=False):
    '''
    Multiplication using SC AND gate
    
    :param p1: probability input 1
    :param p2: probability input 2
    :param scale_input: scale of input probabilities
    :param bit_length: bit length of the stochastic representation
    :param PPA_mode: metric calculation mode

    :return: output probability after multiplication
    if PPA_mode is True, also return PPA score ratio (larger value is better)
    '''
    # :output p1*p2*scale_input 
    if PPA_mode:
        metric = PPA_stochastic_mult(bit_length)
        PPA_result = (metric['energy'] * metric['area'] * metric['delay']) ** (1/3)
        metric_binary = PPA_binary_mult(bit_length)
        PPA_result_binary = (metric_binary['energy'] * metric_binary['area'] * metric_binary['delay']) ** (1/3)
        return load_lookup_table(mode='AND', p1=p1, p2=p2, scale_input=scale_input, bit_length=bit_length), PPA_result/PPA_result_binary 
    else:
        return load_lookup_table(mode='AND', p1=p1, p2=p2, scale_input=scale_input, bit_length=bit_length)
    

# Example: Stochastic Average Pooling using MUX gate
def sc_avg_pooling(input_matrix, pool_size=2, stride=2, scale_input=1.0, bit_length=256):
    """
    Perform stochastic average pooling using the MUX gate from the library.

    Parameters:
    - input_matrix: 2D NumPy array representing the feature map
    - pool_size: size of the pooling window (default is 2x2)
    - stride: stride of the pooling operation (default is 2)
    - scale_input: scaling factor for input probabilities (default is 1.0)

    Returns:
    - pooled_matrix: the pooled feature map after applying stochastic average pooling
    """
    height, width = input_matrix.shape
    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1
    pooled_matrix = np.zeros((out_height, out_width))

    # input scaling into [0, 1]
    scaled_factor = np.max(input_matrix)
    input_matrix = input_matrix / scaled_factor

    # Iterate over the pooling window
    for i in range(out_height):
        for j in range(out_width):
            # Define the pooling region
            h_start, h_end = i * stride, i * stride + pool_size
            w_start, w_end = j * stride, j * stride + pool_size
            pool_region = input_matrix[h_start:h_end, w_start:w_end]

            # Apply stochastic simulation using MUX gates

            for m in range(pool_size):
                for n in range(pool_size):
                    if m == 0 and n == 0:
                        ps = pool_region[0, 0]
                    else:
                        p2 = pool_region[m, n]
                        # Update p1 using MUX to average with p2
                        ps = load_lookup_table(mode="MUX", p1=ps, p2=p2, scale_input=scale_input, scale_mux=0.5, bit_length=bit_length)  # Average factor 0.5
    
            pooled_matrix[i, j] = ps  # Directly take the result from MUX gate

    # Rescale back to original range
    pooled_matrix = pooled_matrix * scaled_factor
    return pooled_matrix

# Example: Stochastic Convolution using AND and Scaled Adder
def sc_convolution(input_matrix, kernel, stride=1, scale_input=1.0, scale_mux=0.5, bit_length=256):
    """
    Perform stochastic convolution using AND gates for multiplication and scaled adders for accumulation.

    Parameters:
    - input_matrix: 2D NumPy array representing the input feature map
    - kernel: 2D NumPy array representing the convolution kernel
    - stride: stride of the convolution operation (default is 1)
    - scale_input: scaling factor for input probabilities (default is 1.0)
    - scale_mux: scaling factor for the scaled adder MUX gate (default is 0.5)

    Returns:
    - output_matrix: the feature map after applying stochastic convolution
    """
    in_height, in_width = input_matrix.shape
    k_height, k_width = kernel.shape
    out_height = (in_height - k_height) // stride + 1
    out_width = (in_width - k_width) // stride + 1
    output_matrix = np.zeros((out_height, out_width))

    # input scaling into [0, 1]
    scaled_factor = np.max(input_matrix)
    input_matrix = input_matrix / scaled_factor

    # Iterate over the output feature map
    for i in range(out_height):
        for j in range(out_width):
            conv_sum = 0.0
            # Perform convolution operation
            for m in range(k_height):
                for n in range(k_width):
                    p1 = input_matrix[i * stride + m, j * stride + n]
                    p2 = kernel[m, n]
                    # Stochastic multiplication using AND gate
                    mult_result = load_lookup_table(mode="AND", p1=p1, p2=p2, scale_input=scale_input, bit_length=bit_length)
                    # Accumulate using scaled adder (average here)
                    if m == 0 and n == 0:
                        conv_sum = mult_result
                    else:
                        conv_sum = load_lookup_table(mode="MUX", p1=conv_sum, p2=mult_result, scale_input=scale_input, scale_mux=scale_mux, bit_length=bit_length)
            output_matrix[i, j] = conv_sum

    # Rescale back to original range
    output_matrix = output_matrix * scaled_factor
    return output_matrix