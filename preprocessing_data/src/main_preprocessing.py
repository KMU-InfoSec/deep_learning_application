import multiprocessing as mp
import time

from preprocessing.make_idb_ops import *
from preprocessing.make_fh import *

if __name__ == '__main__':
    mp.freeze_support()
    p = mp.Pool(CPU_COUNT)

    print('*'*50)
    start_time = time.time()

    # idb-ops
    # input_file_lists = create_file_list(INPUT_FILE_PATH)
    # print("Total File Count : {}".format(len(input_file_lists)))
    # p.map(make_idb_ops, input_file_lists)
    # print("elapsed time: {}".format(time.time() - start_time))

    # fh
    ops_file_lists = create_file_list(OPS_PATH)
    print("Total OPS Count : {}".format(len(ops_file_lists)))
    p.map(make_fh, ops_file_lists)
    print("elapsed time: {}".format(time.time() - start_time))
    pass
