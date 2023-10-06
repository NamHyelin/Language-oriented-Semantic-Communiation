import random
import numpy as np
from QAM import Pb_qam


def string_2_ascii(string):
  binary_code = []
  for char in string:
      row = []
      for ch in char:
          row.append(bin(ord(ch))[2:].zfill(8))
      binary_code.append(row[0])
  return binary_code



def ascii_2_string(binary_code):
  string = ""
  for binary in binary_code:
      ascii_code = int(binary, 2)
      char = chr(ascii_code)
      string += char
  return string


def introduce_errors(bit_string, p):
    errored_bits = []
    for bit in bit_string:
        if random.random() < p:
            errored_bit = '0' if bit == '1' else '1'
            errored_bits.append(errored_bit)
        else:
            errored_bits.append(bit)
    return ''.join(errored_bits)


def introduce_errors_list(bit_strings, p):
    errored_strings=[]
    for bit_string in bit_strings:
        errored_string = introduce_errors(bit_string, p)
        errored_strings.append(errored_string)
    return errored_strings


def string2len(myarray):
    # converts array of string to array of length of each string
    string_mask = np.array([[isinstance(item, str) for item in row] for row in myarray])
    lengths = np.vectorize(len)(myarray[string_mask])
    result_array = np.zeros_like(myarray)
    result_array[string_mask] = lengths
    result_array = result_array.astype(np.int)
    return result_array


if __name__ == "__main__":
    string = 'frolicking'
    # # set
    # prob=0.005
    # 16-qam
    idx=1
    prob=Pb_qam[idx]

    binary_code = string_2_ascii(string)
    string = ascii_2_string(binary_code)
    print('binary code for {}: {}'.format(string, binary_code))
    print('Converting binary code back to text: ', string)
    code_err= introduce_errors_list(binary_code, p=prob)
    string_err= ascii_2_string(code_err)
    print('Errored text: ', string_err)
