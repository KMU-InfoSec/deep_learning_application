import os
import pathos.pools as pp
import pickle
from sklearn.preprocessing import MinMaxScaler

CPU_COUNT = 8
EXTENSION = 'fh_bovd2'
TRAIN_DATA_PATH = os.path.normpath(r'D:/working_board/toy_dataset')
TEST_DATA_PATH = os.path.normpath(r'D:/working_board/kisa_dataset')


def get_file_list(input_path, ext):
    ret = list()
    for path, _, files in os.walk(input_path):
        for file in files:
            if ext == os.path.splitext(file)[-1][1:]:
                ret.append(os.path.join(path, file))

    return ret


def load(file_path):
    return pickle.load(open(file_path, 'rb'))


def save(file_path, content):
    with open(file_path, 'wb') as f:
        pickle.dump(content, f)


def scale(arr):
    result = MinMaxScaler().fit_transform(arr)

    return result


def main():
    # set path
    print('set path')
    train_file_list = get_file_list(TRAIN_DATA_PATH, EXTENSION)
    test_file_list = get_file_list(TEST_DATA_PATH, EXTENSION)
    total_file_list = train_file_list + test_file_list

    # set data
    print('set data')
    p = pp.ProcessPool(CPU_COUNT)
    total_feature_list = p.map(load, total_file_list)
    print('finish loading data ')

    total_scaled_feature_list = scale(total_feature_list)
    print('finish scaling')

    # save
    for src_path, content in zip(total_file_list, total_scaled_feature_list):
        dst_path = src_path.replace(EXTENSION, EXTENSION+'n')
        dir_name = os.path.dirname(dst_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        save(dst_path, content)
    print('finish saving')


if __name__ == '__main__':
    main()
