import configparser

from model import *
from util import *
from sklearn.model_selection import KFold

# config parameters
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')


def run_WINS(model_dic):
    mal_data, ben_data, indices = split_train_test_data(CLASS_TYPE, mal_path=model_dic['mal_dir'], ben_path=model_dic['ben_dir'],
                                                        mal_train_start_date='20170802', mal_train_end_date='20170831',
                                                        mal_test_start_date='20171001', mal_test_end_date='20171029',
                                                        ben_ratio='4:1', ext=model_dic['fh_type'],
                                                        fhs_flag=model_dic['fhs_flag'])

    print('mal #: {}'.format(len(mal_data)))
    print('ben #: {}'.format(len(ben_data)))

    print('*' * 50)
    print('@ load model')

    model_dic['mal_path'] = mal_data
    model_dic['ben_path'] = ben_data
    model_dic['indices'] = indices

    classifier = KISNet(model_num=step, model_dic=model_dic)
    classifier.train()
    classifier.evaluate()
    pass


def run_KISA(model_dic):
    # data
    train_mal_data = np.array(walk_dir(model_dic['mal_dir'], model_dic['fh_type']))
    train_ben_data = np.array(walk_dir(model_dic['ben_dir'], model_dic['fh_type']))
    test_file_path = config.get('PATH', 'TEST_DIR')
    label_path = model_dic['label_path']
    test_mal_data, test_ben_data = read_kisa_data(test_file_path, label_path, model_dic['fh_type'])

    mal_data = np.concatenate((train_mal_data, test_mal_data))
    ben_data = np.concatenate((train_ben_data, test_ben_data))

    print('mal #: {}'.format(len(mal_data)))
    print('ben #: {}'.format(len(ben_data)))

    # indices
    train_mal_indices = list(np.arange(len(train_mal_data)))
    test_mal_indices = list(np.arange(len(train_mal_data), len(train_mal_data)+len(test_mal_data)))
    train_ben_indices = list(np.arange(len(train_ben_data)))
    test_ben_indices = list(np.arange(len(train_ben_data), len(train_ben_data)+len(test_ben_data)))

    indices = list()
    indices.append((np.asarray(train_mal_indices), np.asarray(test_mal_indices)))
    indices.append((np.asarray(train_ben_indices), np.asarray(test_ben_indices)))

    print('*' * 50)
    print('@ load model')

    model_dic['mal_path'] = mal_data
    model_dic['ben_path'] = ben_data
    model_dic['indices'] = indices

    classifier = KISNet(model_num=step, model_dic=model_dic)
    classifier.train()
    classifier.evaluate()


# K-fold learning, train-test 지정하고 학습하는 것 코드 따로 만들기
def run_using_k_fold(model_dic, once_flag):
    k_fold_value = int(config.get('BASIC_INFO', 'K_FOLD_VALUE'))
    fh_type = model_dic['fh_type']

    mal_data, ben_data = np.array(walk_dir(model_dic['mal_dir'], fh_type)), list()
    print('mal #: {}'.format(len(mal_data)))
    if CLASS_TYPE == 'binary':
        ben_data = np.array(walk_dir(model_dic['ben_dir'], fh_type))
    print('ben #: {}'.format(len(ben_data)))

    cv = KFold(n_splits=k_fold_value, shuffle=True, random_state=0)

    if CLASS_TYPE == 'binary':
        cv_split_obj = zip(cv.split(mal_data), cv.split(ben_data))
    else:
        cv_split_obj = cv.split(mal_data)

    acc_list = list()
    for idx, indices in enumerate(cv_split_obj):
        print('*' * 50)
        print('{}번째 실험'.format(idx+1))
        print('@ load model')

        model_dic['mal_path'] = mal_data
        model_dic['ben_path'] = ben_data
        model_dic['indices'] = indices

        classifier = KISNet(model_num=step, model_dic=model_dic)
        classifier.train()
        acc = classifier.evaluate()
        if once_flag:
            break
        acc_list.append(acc)
    else:
        mean_acc = sum(acc_list)/len(acc_list)
        print('평균 정확도: {}'.format(mean_acc))
    pass


if __name__ == '__main__':
    step = int(config.get('BASIC_INFO', 'MODEL_STEP'))
    net_type = config.get('CLASSIFIER', 'NETWORK')
    print('[{0}] step {1}'.format(net_type, step))

    # check classification type
    if int(config.get('CLASSIFIER', 'OUTPUT_SIZE')) == 2:
        CLASS_TYPE = 'binary'
    else:
        CLASS_TYPE = 'multiclass'
    print('@ {} CLASSIFICATION'.format(CLASS_TYPE))

    model_arg = {
        'LR_DECAY_OPTION': config.get('CLASSIFIER', 'LR_DECAY_OPTION'),
        'LR_DECAY_EPOCH': int(config.get('CLASSIFIER', 'LR_DECAY_EPOCH')),
        'LR_DECAY_DROP_RATIO': float(config.get('CLASSIFIER', 'LR_DECAY_DROP_RATIO')),
        'L2_REGULARIZATION': config.get('CLASSIFIER', 'L2_REGULARIZATION'),
        'L2_REGULARIZATION_SCALE': float(config.get('CLASSIFIER', 'L2_REGULARIZATION_SCALE')),
        'batch_size': int(config.get('CLASSIFIER', 'BATCH_SIZE')),
        'ben_dir': os.path.normpath(config.get('PATH', 'BEN_DIR')),
        'class_type': CLASS_TYPE,
        'data_type': config.get('BASIC_INFO', 'DATA_TYPE'),
        'epoch': int(config.get('CLASSIFIER', 'EPOCH')),
        'fh_type': config.get('BASIC_INFO', 'INPUT_FILE_TYPE'),
        'gpu_num': int(config.get('CLASSIFIER', 'GPU_NUM')),
        'keep_prob': float(1 - float(config.get('CLASSIFIER', 'DROPOUT_PROB'))),
        'label_path': config.get('PATH', 'LABEL_FILE'),
        'learning_rate': float(config.get('CLASSIFIER', 'LEARNING_RATE')),
        'model_storage': config.get('CLASSIFIER', 'MODEL_STORAGE'),
        'model_network': config.get('CLASSIFIER', 'NETWORK'),
        'net_input_size': int(config.get('CLASSIFIER', 'INPUT_SIZE')),
        'net_output_size': int(config.get('CLASSIFIER', 'OUTPUT_SIZE')),
        'net_type': config.get('CLASSIFIER', 'NETWORK'),
        'mal_dir': os.path.normpath(config.get('PATH', 'MAL_DIR'))
    }

    if model_arg['data_type'] == 'WINS':
        run_WINS(model_arg)
    elif model_arg['data_type'] == 'KISA':
        run_KISA(model_arg)
    else:
        run_using_k_fold(model_arg, once_flag=int(config.get('BASIC_INFO', 'KFOLD_ONCE_FLAG')))
    pass
