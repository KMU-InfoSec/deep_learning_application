import configparser
from data import *
import datetime
import itertools
import hashlib
import matplotlib.pyplot as plt
import os
import _pickle
import random
from sklearn.metrics import confusion_matrix


# config parameters
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')


# function: 입력 파일의 이름을 md5로 바꿔주는 함수
def convert_filename_to_md5(input_path):
    cnt = 0
    for path, _, files in os.walk(input_path):
        for file in files:
            file_md5 = get_md5(os.path.join(path, file))
            os.rename(os.path.join(path, file), os.path.join(path, file_md5+'.vir'))
            cnt += 1
            if cnt % 200 == 0:
                print(cnt)
    pass


# function: virussign 의 vir 파일을 fixed size vector(확장자: bin)로 변환하는 함수
# up-sampling / down-sampling 기술이 적용된다.
def convert_vir_to_bin(input_dir):
    print('conveert vir to bin start')
    fixed_file_size = int(16384)

    file_lists = os.listdir(input_dir)
    for file in file_lists:
        file_name, ext = os.path.splitext(file)
        # extension check
        if ext != '.vir':
            continue

        content = list(open(os.path.join(input_dir, file), 'rb').read())
        file_size = len(content)

        if file_size > fixed_file_size:  # down-sampling
            rest = file_size % fixed_file_size
            if rest == 0:
                convert_list = np.array(content)
            else:
                zero_padding_list = [0 for _ in range(rest, fixed_file_size)]
                convert_list = np.array(content + zero_padding_list)
            result_list = np.average(convert_list.reshape(len(convert_list)//fixed_file_size, fixed_file_size), axis=0)
        elif file_size < fixed_file_size:  # up-sampling
            blank_size = fixed_file_size - file_size
            for i in range(blank_size):
                content.append(content[i])
            result_list = np.array(content)
            pass
        else:
            result_list = np.array(content)
            pass

        result_list = [x / 255 for x in result_list]  # normalization to 255(max value)

        dst_path = input_dir.replace('raw', 'bin')

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        with open(os.path.join(dst_path, file_name+'.bin'), 'wb') as f:
            _pickle.dump(result_list, f)

    print('{} end'.format(input_dir))
    pass


# function: 입력 디렉토리가 존재하는지 확인하는 함수
def check_existing_dir(file_path):
    if not os.path.exists(os.path.normpath(os.path.abspath(file_path))):
        try:
            os.makedirs(file_path)
        except:
            pass
    pass


# function: 입력 파일에 대한 md5 hash value를 16진수로 반환하는 함수
def get_md5(path, block_size=8192):
    with open(path, 'rb') as f:
        hasher = hashlib.md5()
        buf = f.read(block_size)
        while buf:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()


# function: 시작일과 종료일 사이에 있는 모든 날짜를 반환하는 함수
def get_range_dates(start_date, end_date):
    date_form = '%Y%m%d'
    s = datetime.datetime.strptime(start_date, date_form).date()
    d = datetime.datetime.strptime(end_date, date_form).date()

    result_date = set()
    for day in range((d-s).days + 1):
        result_date.add((s + datetime.timedelta(days=day)).strftime(date_form))

    return result_date


# function: 학습 모델의 예측 결과와 ground truth를 입력으로 받아 혼동 행렬를 그려주는 함수
def plot_confusion_matrix(step, y_true, y_pred, output_size):
    # check result directory
    result_dir = 'result'
    check_existing_dir(result_dir)

    print('plot confusion matrix start: ', end='')

    # preprocessing
    y_true = [x for x in y_true if x != -1]
    y_pred = [x for x in y_pred if x != -1]

    # compute confusion matrix
    cnf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # configuration
    np.set_printoptions(precision=2)

    if output_size == 2:
        labels = ['benign', 'malware']
    else:
        # toy dataset label
        # labels = ['Virus', 'Worm', 'Trojan', 'not-a-virus:Downloader', 'Trojan-Ransom', 'Backdoor']
        labels = list(range(output_size))
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    norm_flag = True
    plot_title = 'Confusion matrix'
    cmap = plt.cm.Blues

    if norm_flag:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    for row in cnf_matrix:
        for val in row:
            print('{0:.2f}'.format(val), end=' ')
        print()

    # plotting start
    plt.figure()
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(plot_title)
    plt.colorbar()

    # information about each block's value
    fmt = '.3f' if norm_flag else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    # insert legend information
    # import matplotlib.patches as mpatches
    # patches = [mpatches.Patch(color='white', label='G{num} = {group}'.format(num=i+1, group=labels[i])) for i in range(len(labels))]
    # plt.legend(handles=patches, bbox_to_anchor=(-0.60, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig(os.path.join(result_dir, 'conf_matrix{}'.format(step)))
    print('--plot confusion matrix finish--')
    pass


def read_test_data(test_file_path, label_path, fh_type):
    label_dict = dict()
    with open(label_path, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for line in rdr:
            file_name, label = line[0], int(line[1])
            label_dict[file_name] = label

    mal_data_path = list()
    ben_data_path = list()
    for (file_name, label) in label_dict.items():
        full_path = os.path.join(test_file_path, file_name + '.{}'.format(fh_type))
        if os.path.exists(full_path):
            if label == 0:  # benignware
                ben_data_path.append(full_path)
            else:
                mal_data_path.append(full_path)

    return np.array(mal_data_path), np.array(ben_data_path)


# function:
def save_result_to_csv(step, filenames, actuals, preds, mal_probs):
    # check result directory
    result_dir = 'result'
    check_existing_dir(result_dir)

    # delete upper directory
    filenames = [filename.split(os.sep)[-1] for filename in filenames]

    # save result as csv file to SEEK
    with open(os.path.join(result_dir, 'inference{}.csv'.format(step)), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['md5', 'label', 'prob'])  # 확률
        for name, pred_label, actual_label, mal_prob in zip(filenames, preds, actuals, mal_probs):
            wr.writerow([name, actual_label, '{0:.4f}'.format(mal_prob)])
        else:  # 딥러닝이 맞추지 못한 걸 csv file에 추가해야 한다.
            # load test file names
            test_file_md5_dict = dict()
            with open(config.get('PATH', 'LABEL_FILE'), 'r', encoding='utf-8') as f:
                for line in csv.reader(f):
                    md5, label = line[0].replace('.vir', ''), int(line[1])
                    test_file_md5_dict[md5] = label
            for md5 in test_file_md5_dict:
                if not md5 in filenames:
                    wr.writerow([md5, test_file_md5_dict[md5], 0.5])  # 확률

    # save result that gets wrong cases as csv file
    # with open(os.path.join(result_dir, 'profiling{}.csv'.format(step)), 'w', newline='', encoding='utf-8') as f:
    #     wr = csv.writer(f)
    #     for name, actual_label, pred_label in zip(filenames, actuals, preds):
    #         if actual_label != pred_label:
    #             wr.writerow([name, actual_label, pred_label])
    pass


# function: 악성/정상 파일을 train/test 데이터셋으로 분류하는 함수. 특별히 악성코드는 수집날짜로 분류한다.
def split_train_test_data_to_date(class_type, mal_path, ben_path, mal_train_start_date, mal_train_end_date,
                          mal_test_start_date, mal_test_end_date, ben_ratio, ext, fhs_flag):
    print('@ split train test data')

    # 악성코드 경로를 받아오고, 날짜에 따라 학습/테스트 셋을 분류
    mal_data = np.array(walk_dir(os.path.join(mal_path, 'fhs') if fhs_flag else mal_path, 'fhs' if fhs_flag else ext))
    mal_train_dates = get_range_dates(mal_train_start_date, mal_train_end_date)
    mal_test_dates = get_range_dates(mal_test_start_date, mal_test_end_date)
    mal_train_indices, mal_test_indices = list(), list()

    for cnt, data in enumerate(mal_data):
        file_name = os.path.splitext(os.path.basename(data))[0]
        upper_path = data.split(os.sep)[-2]
        if (file_name in mal_train_dates) or (upper_path in mal_train_dates):
            print('train: {}'.format(file_name))
            mal_train_indices.append(cnt)
        elif (file_name in mal_test_dates) or (upper_path in mal_test_dates):
            print('test: {}'.format(file_name))
            mal_test_indices.append(cnt)

    # 정상파일 경로를 받아오고, 비율에 따라 학습/테스트 셋을 분류
    ben_data = np.array(walk_dir(ben_path, ext)) if class_type == 'binary' else list()
    ben_total_indices = np.arange(len(ben_data)); random.shuffle(ben_total_indices)
    ben_ratio_a, ben_ratio_b = ben_ratio.split(':')
    ben_ratio_number = int(ben_ratio_a)/(int(ben_ratio_a)+int(ben_ratio_b))
    no_ben_train_data = int(ben_ratio_number*len(ben_data))
    ben_train_indices, ben_test_indices = ben_total_indices[:no_ben_train_data], ben_total_indices[no_ben_train_data:]

    indices = list()
    indices.append((np.asarray(mal_train_indices), np.asarray(mal_test_indices)))
    if class_type == 'binary':
        indices.append((ben_train_indices, ben_test_indices))

    return mal_data, ben_data, [indices]


# function: 입력 경로에 대한 모든 파일 경로를 리스트로 반환하는 함수
def walk_dir(input_path, ext):
    print('@ walk dir start')
    result = list()
    for path, dirs, files in os.walk(input_path):
        if len(dirs) == 0:
            print(path)
            for file in files:
                if ext == os.path.splitext(file)[-1][1:]:
                    file_path = os.path.join(path, file)  # store "file path"
                    result.append(file_path)
    print('@ walk dir finish')
    return result
