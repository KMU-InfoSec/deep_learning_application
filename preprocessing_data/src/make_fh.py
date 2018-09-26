import hashlib
import pickle

from preprocessing.settings import *

OPS_TYPE, GRAM_TYPE, VALUE_TYPE = FH_TYPE[3], FH_TYPE[5], FH_TYPE[6]


def exist_dirs(file_path):
    if not os.path.exists(file_path):
        try:
            os.makedirs(file_path)
        except:
            pass


def function_xor_generator(input_list):
    for func_list in input_list:
        func_value = 0
        for bb_list in func_list:
            value = ''.join(bb_list)
            hash_value = int(hashlib.sha256(value.encode('utf-8')).hexdigest(), 16)
            func_value ^= hash_value
        yield func_value  # 10진수


def basic_block_generator(input_list):
    # if OPS_TYPE == 'f':  # fops
    for func_list in input_list:
        for bb_list in func_list:
            yield bb_list


def get_vector_index(window, hash_value=None):
    if hash_value is None:
        # get the hash value
        cryp_hash = int(hashlib.sha256(window.encode('utf-8')).hexdigest(), 16)
        hash_value = cryp_hash

    cryp_hash2 = hash_value // MAX_VECTOR_SIZE
    cryp_hash2 = hash_value >> MAX_VECTOR_SIZE_BIT

    index = hash_value % MAX_VECTOR_SIZE  # can use bit-wise operation
    decision_cnt = cryp_hash2 % 2
    decision_content = cryp_hash2 % FH_CONTENT_BOUNDARY

    return index, decision_cnt, decision_content


def apply_feature_value(vector, index, decision_cnt, decision_ctt):
    if VALUE_TYPE == 'r':
        if decision_cnt == 1:
            vector[index] += 1
        else:  # decision == 0
            vector[index] -= 1
    else:  # content
        vector[index] = max([vector[index], decision_ctt])
        pass


def normalize_vector(vector):
    max_value = max(vector)
    min_value = min(vector)
    if max_value - min_value == 0:
        vector = [0 for _ in range(MAX_VECTOR_SIZE)]
    else:
        vector = [2 * ((x - min_value) / (max_value - min_value)) - 1 for x in vector]

    return vector


def make_fh(file_path):
    # define file path
    file_name, ext = os.path.splitext(os.path.split(file_path)[-1])
    file_name = file_name + '.{}'.format(FH_TYPE)
    sub_file_path = file_path.replace(OPS_PATH, '').replace(os.path.basename(file_path), '')

    save_path = FH_PATH + sub_file_path

    # check directory
    exist_dirs(save_path)

    # if the file was created
    if os.path.exists(os.path.join(save_path, file_name)):
        print('{} existed'.format(file_name))
        return

    # make feature hashing vector

    # step 1. load ops file (pickle format)
    try:
        with open(file_path, 'rb') as f:
            opcodes = pickle.load(f)
        if len(opcodes) == 0:
            print('@ null ops: {}'.format(file_name))
            return
    except:
        print('@ not pickled: {}'.format(file_name))
        return

    # step 2. initialize feature vector
    fh_vector = [0 for _ in range(MAX_VECTOR_SIZE)]

    # step 3. check GRAM_TYPE
    if GRAM_TYPE == 'n':
        # step 3-0. convert 3-d list to 1-d list
        opcodes = [e for sl in opcodes for e in sl]
        opcodes = [e for sl in opcodes for e in sl]
        # step 3-1. check length of opcode list
        count_of_opcode = len(opcodes)
        if count_of_opcode < N_GRAM:
            return

        # step 3-2. initialize sliding window
        window = ''
        for i in range(N_GRAM):
            window += opcodes[i]

        # step 3-3. get the hash value
        index, decision_cnt, decision_ctt = get_vector_index(window)

        # step 3-4. apply the value to the vector
        apply_feature_value(fh_vector, index, decision_cnt, decision_ctt)

        # step 3-5. iterate previous step
        for i in range(N_GRAM, count_of_opcode):
            # step 3-5-2. get window
            window = window[2:] + opcodes[i]

            # step 3-5-3. get the hash value
            index, decision_cnt, decision_ctt = get_vector_index(window)

            # step 3-5-4. apply the value to the vector
            apply_feature_value(fh_vector, index, decision_cnt, decision_ctt)
    # ------------------------------------------------------------------------------------- #
    elif GRAM_TYPE == 'v':  # fops, bops
        if OPS_TYPE == 'b':
            for opcode in basic_block_generator(opcodes):
                # step 3-1. get window (variable size)
                window = ''.join(opcode)

                # step 3-2. get the hash value
                index, decision_cnt, decision_ctt = get_vector_index(window)

                # step 3-3. apply the value to the vector
                apply_feature_value(fh_vector, index, decision_cnt, decision_ctt)
        elif OPS_TYPE == 'f':
            for func_xor_value in function_xor_generator(opcodes):
                # step 3-2. get the index, values
                index, decision_cnt, decision_ctt = get_vector_index('', func_xor_value)

                # step 3-3. apply the value to the vector
                apply_feature_value(fh_vector, index, decision_cnt, decision_ctt)

    else:
        print('invalid GRAM TYPE!')
        return

    # step 4. scaling(max-min normalization to range [-1, 1])
    fh_vector = normalize_vector(fh_vector)

    # save file
    with open(os.path.join(save_path, file_name), 'wb') as f:
        pickle.dump(fh_vector, f)

    print('{} fh finished'.format(file_name))
    pass


if __name__ == '__main__':
    # _base_path = 'D:\\working_board\\toy_dataset\\malware\\ops\\00a2caa5de1a55c99a9c16164123109d.ops'
    #
    # make_fh(_base_path)
    pass

