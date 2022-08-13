from unittest import result
from model.template import *

if __name__ == '__main__':
    result_data, max_idx = cross([2,32,2], [1,5,1], 0.01, 0.01, 1000)
    print(max_idx)
    print(result_data[0][1])