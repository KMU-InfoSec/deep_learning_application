from model import *
from util import *
from sklearn.model_selection import KFold

# config parameters
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')


def get_data_path(model_arg):
    input_option_type = model_arg['input_option_type']
    # config.ini 의 [BASIC_INFO]->'INPUT_OPTION_TYPE' 설명 참고
    # 1: [MAL_DIR,BEN_DIR]에 대해 [K-FOLD CROSS VALIDATION]을 사용. [LABEL_FILE]은 [MAL_DIR]의 레이블 저장.
    # 2: [MAL_DIR,BEN_DIR]은 [학습 데이터]로 사용하고, [TEST_DIR]은 [테스트 데이터]로 사용. [LABEL_FILE]은 [TEST_DIR]의 레이블 저장.
    # 3: [MAL_DIR,BEN_DIR]에 대해 학습/테스트를 진행. [MAL_DIR]은 [날짜] 기준으로, [BEN_DIR]은 [비율] 기준으로 학습/테스트 데이터를 분할.
    if input_option_type == '1':
        # data
        mal_data = np.array(walk_dir(model_arg['mal_dir'], model_arg['input_file_type']))
        ben_data = np.array(walk_dir(model_arg['ben_dir'], model_arg['input_file_type']) if model_arg['class_type'] == 'binary' else list())

        # index
        cv = KFold(n_splits=model_arg['k_fold_value'], shuffle=True, random_state=0)
        cv_mal_obj = cv.split(mal_data)
        cv_ben_obj = cv.split(ben_data) if model_arg['class_type'] == 'binary' else np.asarray([list() for _ in range(model_arg['k_fold_value'])])
        indices_obj = zip(cv_mal_obj, cv_ben_obj)
    elif input_option_type == '2':
        # data
        train_mal_data = np.array(walk_dir(model_arg['mal_dir'], model_arg['input_file_type']))
        train_ben_data = np.array(walk_dir(model_arg['ben_dir'], model_arg['input_file_type']))
        test_mal_data, test_ben_data = read_test_data(model_arg['test_dir'], model_arg['label_path'], model_arg['input_file_type'])

        mal_data = np.concatenate((train_mal_data, test_mal_data))
        ben_data = np.concatenate((train_ben_data, test_ben_data))

        # index
        train_mal_indices = list(np.arange(len(train_mal_data)))
        test_mal_indices = list(np.arange(len(train_mal_data), len(train_mal_data) + len(test_mal_data)))
        train_ben_indices = list(np.arange(len(train_ben_data)))
        test_ben_indices = list(np.arange(len(train_ben_data), len(train_ben_data) + len(test_ben_data)))

        indices = list()
        indices.append((np.asarray(train_mal_indices), np.asarray(test_mal_indices)))
        indices.append((np.asarray(train_ben_indices), np.asarray(test_ben_indices)))
        indices_obj = [indices]
    elif input_option_type == '3':
        # data, index
        mal_data, ben_data, indices_obj = split_train_test_data_to_date(class_type=model_arg['class_type'],
                                                        mal_path=model_arg['mal_dir'], ben_path=model_arg['ben_dir'],
                                                        mal_train_start_date='20170802', mal_train_end_date='20170930',
                                                        mal_test_start_date='20171001', mal_test_end_date='20171029',
                                                        ben_ratio='4:1', ext=model_arg['input_file_type'],
                                                        fhs_flag=model_arg['fhs_flag'])
    else:
        raise NotImplementedError

    return mal_data, ben_data, indices_obj


def main():
    model_arg = {
        'batch_size': int(config.get('CLASSIFIER', 'BATCH_SIZE')),
        'ben_dir': os.path.normpath(config.get('PATH', 'BEN_DIR')),
        'class_type': 'binary' if int(config.get('CLASSIFIER', 'OUTPUT_SIZE')) == 2 else 'multiclass',
        'epoch': int(config.get('CLASSIFIER', 'EPOCH')),
        'fhs_flag': int(config.get('BASIC_INFO', 'FHS_FLAG')),
        'gpu_num': int(config.get('CLASSIFIER', 'GPU_NUM')),
        'input_file_type': config.get('BASIC_INFO', 'INPUT_FILE_TYPE'),
        'input_option_type': config.get('BASIC_INFO', 'INPUT_OPTION'),
        'keep_prob': float(1 - float(config.get('CLASSIFIER', 'DROPOUT_PROB'))),
        'k_fold_once_flag': int(config.get('BASIC_INFO', 'K_FOLD_ONCE_FLAG')),
        'k_fold_value': int(config.get('BASIC_INFO', 'K_FOLD_VALUE')),
        'label_path': config.get('PATH', 'LABEL_FILE'),
        'learning_rate': float(config.get('CLASSIFIER', 'LEARNING_RATE')),
        'l2_reg_scale': float(config.get('CLASSIFIER', 'L2_REG_SCALE')),
        'l2_reg_option': config.get('CLASSIFIER', 'L2_REG_OPTION'),
        'lr_decay_drop_ratio': float(config.get('CLASSIFIER', 'LR_DECAY_DROP_RATIO')),
        'lr_decay_epoch': int(config.get('CLASSIFIER', 'LR_DECAY_EPOCH')),
        'lr_decay_option': config.get('CLASSIFIER', 'LR_DECAY_OPTION'),
        'mal_dir': os.path.normpath(config.get('PATH', 'MAL_DIR')),
        'model_storage': config.get('CLASSIFIER', 'MODEL_STORAGE'),
        'model_network': config.get('CLASSIFIER', 'NETWORK'),
        'net_input_size': int(config.get('CLASSIFIER', 'INPUT_SIZE')),
        'net_output_size': int(config.get('CLASSIFIER', 'OUTPUT_SIZE')),
        'net_type': config.get('CLASSIFIER', 'NETWORK'),
        'test_dir': config.get('PATH', 'TEST_DIR')
    }

    print('@ [{0}] step {1}'.format(model_arg['net_type'], model_arg['gpu_num']))
    print('@ {} CLASSIFICATION'.format(model_arg['class_type']))

    model_arg['mal_path'], model_arg['ben_path'], indices_obj = get_data_path(model_arg)

    print('mal #: {}'.format(len(model_arg['mal_path'])))
    print('ben #: {}'.format(len(model_arg['ben_path'])))

    acc_list = list()
    for idx, indices in enumerate(indices_obj):
        print('*' * 50)
        print('{}번째 실험'.format(idx + 1))
        print('@ load model')

        model_arg['indices'] = indices

        model = KISNet(model_num=model_arg['gpu_num'], model_arg=model_arg, model_reuse_flag=False)
        model.train()
        acc = model.evaluate()
        acc_list.append(acc)
        if model_arg['k_fold_once_flag']:
            break
    print('평균 정확도: {}'.format(sum(acc_list)/len(acc_list)))


if __name__ == '__main__':
    main()
