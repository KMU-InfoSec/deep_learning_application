import pathos.pools as pp
import time

from make_idb_ops import *
import make_fh


def disassemble(p):
    print('*' * 50)
    start_time = time.time()

    file_list = create_file_list(INPUT_FILE_PATH)
    print("Total File Count : {}".format(len(file_list)))
    feature_list = p.map(make_idb_ops, file_list)
    print("elapsed time: {}".format(time.time() - start_time))

    return feature_list


def make_feature_vector(p):
    print('*' * 50)
    start_time = time.time()

    file_list = create_file_list(FH_INPUT_PATH)
    print("Total File Count : {}".format(len(file_list)))
    ret = p.map(make_fh.main, file_list)
    print("elapsed time: {}".format(time.time() - start_time))

    return ret


if __name__ == '__main__':
    p = pp.ProcessPool(CPU_COUNT)

    disassemble(p)
    # make_feature_vector(p)

    print('{0} {1} {2} {3}'.format(FILE_CLASS, FH_TYPE, N_GRAM, MAX_VECTOR_SIZE))
    pass
