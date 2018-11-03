import hashlib
import pickle

from settings import *

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
        hash_value = int(hashlib.sha256(window.encode('utf-8')).hexdigest(), 16)

    # hash value 가 두 개 필요하므로 해시함수의 값을 나눈다. hash value --> (hash1, hash2)
    hash_1 = hash_value >> MAX_VECTOR_SIZE_BIT
    hash_2 = hash_value & ((1 << MAX_VECTOR_SIZE_BIT) - 1)

    index = hash_2
    decision_sign = hash_1 & 0x1
    decision_content = hash_1 & (FH_CONTENT_BOUNDARY - 1)

    return index, decision_sign, decision_content


def apply_feature_value(vector, index, decision_sign, decision_content):
    if VALUE_TYPE == 'r':
        if decision_sign == 1:
            vector[index] += 1
        else:  # decision == 0
            vector[index] -= 1
    else:  # content
        vector[index] = max([vector[index], decision_content])
        pass


def normalize_vector(vector):
    max_value = max(vector)
    min_value = min(vector)

    if min_value == max_value:  # 0-vector
        vector = [0] * MAX_VECTOR_SIZE
    else:
        vector = [2 * ((x - min_value) / (max_value - min_value)) - 1 for x in vector]

    return vector


def analyze(file_name, ops_set, fh_ops_map, fh_vector):
    log = ''

    log += '----------------------\n'
    log += '1. {}에 출현한 서로 다른 basic block ops 개수: {}개\n'.format(file_name, len(ops_set))
    dot_flag = True
    log += '>> '
    for i, bb in enumerate(ops_set):
        if i <= 3 or len(ops_set) - i <= 3:
            log += '{} '.format(bb)
        else:
            if dot_flag:
                log += '... '
                dot_flag = False
            else:
                continue
    else:
        log += '\n'
    log += '----------------------\n\n'

    log += '----------------------\n'
    log += '2. 서로 다른 basic block ops 가 피처 벡터에 매핑된 결과\n'
    dot_flag = True
    size = len(fh_vector)
    for idx, (key, value) in enumerate(fh_ops_map.items()):
        if idx <= 4 or size-idx <= 4:
            log += ' {}th: '.format(idx)
            for ops in value:
                log += '{}, '.format(ops)
            log += '\n'
        else:
            if dot_flag:
                log += '... '
                dot_flag = False
            else:
                continue
    log += '----------------------\n\n'

    log += '----------------------\n'
    log += '3. 피처 벡터의 원소 개수: {}개\n'.format(size)
    log += '>> '
    dot_flag = True
    for i in range(size):
        if i <= 3 or size-i <= 3:
            log += '{} '.format(fh_vector[i])
        else:
            if dot_flag:
                log += '... '
                dot_flag = False
            else:
                continue
    else:
        log += '\n'

    log += '----------------------\n'

    with open('feature_info.txt', 'w') as f:
        for line in log:
            f.write(line)
    pass


def make_fh(file_path):
    # define file path
    file_name, ext = os.path.splitext(os.path.split(file_path)[-1])
    file_name = file_name + '.{}'.format(FH_TYPE)
    sub_file_path = file_path.replace(FH_INPUT_PATH, '').replace(os.path.basename(file_path), '')

    save_path = FH_PATH + sub_file_path

    # check directory
    exist_dirs(save_path)

    # if the file was created
    if SAVE_FH_FLAG:
        if os.path.exists(os.path.join(save_path, file_name)):
            print('{} existed'.format(file_name))
            return

    # make feature hashing vector
    ops_set = set()

    # 성능 검증을 위한 집합 자료형
    analysis_flag = ANALYSIS_FLAG
    if analysis_flag:
        ops_set = set()
        fh_ops_map = dict()  # 어떤 ops가 어떤 index에 맵핑되었는지 확인
        for i in range(MAX_VECTOR_SIZE):
            fh_ops_map[i] = set()

    # step 1. load input file (pickle format)
    try:
        with open(file_path, 'rb') as f:
            content = pickle.load(f)
        if len(content) == 0:
            print('@ null ops: {}'.format(file_name))
            return
    except:
        print('@ not pickled: {}'.format(file_name))
        return

    # step 2. initialize feature vector
    fh_vector = [0] * MAX_VECTOR_SIZE

    # step 3. check GRAM_TYPE
    if GRAM_TYPE == 'n':
        # step 3-0. convert 3-d list to 1-d list
        def make_1d_list(elements):
            if len(elements) == 0:
                return list()
            else:
                ret = list()
                for each in elements:
                    if isinstance(each, list):
                        ret.extend(make_1d_list(each))
                    else:
                        ret.append(each)
            return ret
        content = make_1d_list(content)

        # step 3-1. check length of content
        no_content = len(content)
        if no_content < N_GRAM:
            return

        # step 3-2. initialize sliding window
        window = ''
        for i in range(N_GRAM):
            window += content[i]

        # step 3-3. get the hash value
        index, decision_sign, decision_content = get_vector_index(window)

        # step 3-4. apply the value to the vector
        apply_feature_value(fh_vector, index, decision_sign, decision_content)

        # step 3-5. iterate previous step
        for i in range(N_GRAM, no_content):
            # step 3-5-2. get window
            window = window[len(content[i-N_GRAM]):] + content[i]
            # step 3-5-3. get the hash value
            index, decision_sign, decision_content = get_vector_index(window)
            # step 3-5-4. apply the value to the vector
            apply_feature_value(fh_vector, index, decision_sign, decision_content)
    # ------------------------------------------------------------------------------------- #
    elif GRAM_TYPE == 'v':  # fops, bops
        if OPS_TYPE == 'b':
            for opcode in basic_block_generator(content):
                # step 3-1. get window (variable size)
                window = ''.join(opcode)

                # step 3-2. get the hash value
                index, decision_sign, decision_content = get_vector_index(window)

                # step 3-3. apply the value to the vector
                apply_feature_value(fh_vector, index, decision_sign, decision_content)

                if analysis_flag:
                    ops_set.add(window)
                    fh_ops_map[index].add(window)

        elif OPS_TYPE == 'f':
            for func_xor_value in function_xor_generator(content):
                # step 3-2. get the index, values
                index, decision_sign, decision_content = get_vector_index('', func_xor_value)

                # step 3-3. apply the value to the vector
                apply_feature_value(fh_vector, index, decision_sign, decision_content)
        else:
            raise NotImplementedError
    else:
        print('invalid GRAM TYPE!')
        return

    # step 4. scaling(max-min normalization to range [-1, 1])
    fh_vector = normalize_vector(fh_vector)

    # make log file
    if analysis_flag:
        analyze(file_name, ops_set, fh_ops_map, fh_vector)

    # save file
    if SAVE_FH_FLAG:
        with open(os.path.join(save_path, file_name), 'wb') as f:
            pickle.dump(fh_vector, f)

    print('{0} fh finished'.format(file_name))
    return 0


if __name__ == '__main__':
    _base_path = r'D:\working_board\dataset_kisa\benignware\str\strings_ahnlab\0000b97b3322e5792f8c88e01b4f4313.str'

    make_fh(_base_path)
    pass

