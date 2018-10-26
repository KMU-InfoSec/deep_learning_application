import pathos.pools as pp
import time

from make_idb_ops import *
from make_fh import *


if __name__ == '__main__':
    p = pp.ProcessPool(CPU_COUNT)

    print('*'*50)
    start_time = time.time()

    # idb-ops
    # input_file_lists = create_file_list(INPUT_FILE_PATH)
    # print("Total File Count : {}".format(len(input_file_lists)))
    # p.map(make_idb_ops, input_file_lists)
    # print("elapsed time: {}".format(time.time() - start_time))

    # fh
    # ops_file_list = create_file_list(OPS_PATH)
    # print("Total OPS Count : {}".format(len(ops_file_list)))
    # p.map(make_fh, ops_file_list)
    # print("elapsed time: {}".format(time.time() - start_time))

    # acs fh
    acs_file_list = create_file_list(ACS_PATH)
    print("Total ACS Count : {}".format(len(acs_file_list)))
    p.map(make_fh, acs_file_list)
    print("elapsed time: {}".format(time.time() - start_time))

    pass
