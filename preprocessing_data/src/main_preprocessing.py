import pathos.pools as pp
import time

from make_fh_kisa import *


if __name__ == '__main__':
    p = pp.ProcessPool(CPU_COUNT)

    print('*'*50)
    start_time = time.time()

    file_list = create_file_list(FH_INPUT_PATH)
    print("Total File Count : {}".format(len(file_list)))
    feature_list = p.map(make_fh_kisa, file_list)
    print("elapsed time: {}".format(time.time() - start_time))
    print('{0} {1} {2} {3}'.format(FILE_CLASS, FH_TYPE, N_GRAM, MAX_VECTOR_SIZE))
    pass
