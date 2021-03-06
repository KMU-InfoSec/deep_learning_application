import hashlib
import json
import pickle

from settings import *

OPS_TYPE, GRAM_TYPE, VALUE_TYPE = FH_TYPE[3], FH_TYPE[5], FH_TYPE[6:]


def init_vector():
    if VALUE_TYPE == 'd2':
        return [[0, 0]] * MAX_VECTOR_SIZE
    else:
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


def get_hash_index(hash_digest):
    new_hash_digest = hash_digest >> MAX_VECTOR_SIZE_BIT
    index = hash_digest & ((1 << MAX_VECTOR_SIZE_BIT) - 1)

    return index, new_hash_digest


def update_feature_vector(vector, index, hash_digest, md5=None, doc_info=None, feature=None):
    if VALUE_TYPE == 'r':  # frequent
        decision = hash_digest & 0x1
        value = 1 if decision else -1
        vector[index] += value
    elif VALUE_TYPE == 'c':  # content
        value = hash_digest & (FH_CONTENT_BOUNDARY - 1)
        vector[index] = max([vector[index], value])
    elif VALUE_TYPE == 't':  # tf
        decision = hash_digest & 0x1
        # file_class = FILE_CLASS[:3]
        try:
            tf_value = doc_info['mal'][md5][feature] + doc_info['ben'][md5][feature]  # todo
            value = tf_value if decision else -tf_value
        except:
            value = 0
        vector[index] += value
    elif VALUE_TYPE == 'd':  # df
        decision = hash_digest & 0x1
        file_class = FILE_CLASS[:3]
        df_value = doc_info[file_class].get(feature, 0)
        value = df_value if decision else -df_value
        vector[index] += value
    elif VALUE_TYPE == 'd2':
        decision = hash_digest & 0x1
        df_value = [doc_info['mal'].get(feature, 0), doc_info['ben'].get(feature, 0)]
        value = df_value if decision else [-v for v in df_value]

        vector[index] = [a + b for a, b in zip(vector[index], value)]
    else:
        raise NotImplementedError


def main(file_path, doc_info=None):
    try:
        file_name, save_path = get_file_info(file_path)

        # if check_file(save_path):
        #     return

        feature_list = load(file_path)

        feature_vector = init_vector()
        for feature in set(feature_list):  # 중복 삭제
            hash_digest = int(hashlib.sha256(feature.encode('utf-8')).hexdigest(), 16)

            index, hash_digest = get_hash_index(hash_digest)
            if VALUE_TYPE in ['t', 'd', 'd2']:
                update_feature_vector(feature_vector, index, hash_digest, md5=file_name, doc_info=doc_info, feature=feature)
            else:
                update_feature_vector(feature_vector, index, hash_digest)
        else:
            save(save_path, feature_vector)

        print('{} finished'.format(file_name))

    except Exception as e:
        print('{0} Error: {1}'.format(file_path, e))


if __name__ == '__main__':
    # additional options
    doc_info = None
    if VALUE_TYPE == 't':
        with open(r'./data/tf_label_md5_bb_val.json', 'r') as f:
            doc_info = json.load(f)
    elif VALUE_TYPE in ['d', 'd2']:
        with open(r'./data/df_label_bb_val.json', 'r') as f:
            doc_info = json.load(f)
    else:
        pass

    file_list = create_file_list(FH_INPUT_PATH)

    print("Total File Count : {}".format(len(file_list)))
    for file_path in file_list:
        main(file_path, doc_info)