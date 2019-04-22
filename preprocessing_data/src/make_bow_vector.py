import hashlib
import json
import numpy as np
import operator
import pickle
from sklearn.preprocessing import MinMaxScaler

import warnings
from sklearn.exceptions import DataConversionWarning

from settings import *

OPS_TYPE, GRAM_TYPE, VALUE_TYPE = FH_TYPE[3], FH_TYPE[5], FH_TYPE[6]


def init_vector():
    return [0] * MAX_VECTOR_SIZE


def get_file_info(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_file = file_name + '.{}'.format(FH_TYPE)
    sub_file_path = file_path.replace(FH_INPUT_PATH, '').replace(os.path.basename(file_path), '')
    save_path = FH_PATH + sub_file_path

    # check directory
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    return file_name, os.path.join(save_path, save_file)


def chunk(content):
    ret = list()

    if GRAM_TYPE == 'n':  # n-gram (n=3)
        # -------------------------------------------- #
        def make_1d_list(elements):
            if len(elements) == 0:  # if not iterable
                return list()
            else:
                ret = list()
                for each in elements:
                    if isinstance(each, list):
                        ret.extend(make_1d_list(each))
                    else:
                        ret.append(each)
            return ret
        # -------------------------------------------- #
        content = make_1d_list(content)

        # check the length of content
        no_content = len(content)
        if no_content < N_GRAM:
            return None

        window = str()
        for i in range(0, N_GRAM):
            window += content[i]
        else:
            ret.append(window)

        for i in range(N_GRAM, no_content):
            window = window[len(content[i-N_GRAM]):] + content[i]
            ret.append(window)
        else:
            pass
    elif GRAM_TYPE == 'v':  # v-gram
        if OPS_TYPE == 'b':  # basic block
            for f in content:
                for bb in f:
                    elem = ''.join(bb)
                    if len(elem) >= 2:  # 적어도 하나의 opcode가 존재하면
                        ret.append(elem)
        elif OPS_TYPE == 'f':  # function
            for f in content:
                elem = str()
                for bb in f:
                    elem += ''.join(bb)
                if len(elem) >= 2:
                    ret.append(elem)
            pass
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return ret


def check_file(file_path):
    if os.path.exists(file_path):
        print('{} existed'.format(os.path.basename(file_path)))
        return True
    return False


def load(file_path):
    # load: storage --> memory
    with open(file_path, 'rb') as f:
        content = pickle.load(f)

    return chunk(content)


def save(file_path, content):
    with open(file_path, 'wb') as f:
        pickle.dump(content, f)


def main(file_path, doc_info=None):
    try:
        file_name, save_path = get_file_info(file_path)

        # if check_file(save_path):
        #     return

        feature_list = load(file_path)

        feature_vector = init_vector()
        for feature in set(feature_list):
            try:
                index, value = doc_info[feature][0], doc_info[feature][1]
                feature_vector[index] = value
            except:
                continue

        save(save_path, feature_vector)
        print('{} finished'.format(file_name))
    except Exception as e:
        print('{0} Error: {1}'.format(file_path, e))
    pass


if __name__ == '__main__':
    doc_info = None
    doc_dict = None

    if VALUE_TYPE == 'w':
        with open(r'./data/df_bb_val.json', 'r') as f:
            doc_info = json.load(f)

        # df 내림차순 생성
        doc_list = sorted(doc_info.items(), key=operator.itemgetter(1), reverse=True)[:MAX_VECTOR_SIZE]
        doc_dict = dict()
        for i, each in enumerate(doc_list):
            doc_dict[each[0]] = [i, each[1]]
    else:
        pass

    file_list = create_file_list(FH_INPUT_PATH)
    print("Total File Count : {}".format(len(file_list)))
    for file_path in file_list:
        main(file_path, doc_dict)
