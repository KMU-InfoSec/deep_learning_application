import csv
import numpy as np
import os
import pickle
import random


class DataLoader:
    def __init__(self, mal_paths, ben_paths, label_paths, batch_size, epoch, mode):
        self.class_description = 'PE File Data Loader'
        self.iter_mode = mode  # for mini-batch data feeding
        self.class_type = 'binary' if len(ben_paths) != 0 else 'multiclass'
        print('{}: {} mode'.format(self.class_description, mode))

        # initialize member variable
        self.mal_paths = mal_paths
        self.ben_paths = ben_paths
        self.label_paths = label_paths

        self.file_paths = np.concatenate((self.mal_paths, self.ben_paths), axis=0) if self.class_type == 'binary' else self.mal_paths

        # allocate all data into memory
        print('{} data: set data into memory'.format(mode))
        mal_data = list()
        ben_data = list()
        mal_name_list = list()
        ben_name_list = list()

        # load data: malware
        for i, path in enumerate(self.mal_paths):
            try:
                content = pickle.load(open(path, 'rb'))
            except:
                print('cannot load data: {}'.format(path))
                continue

            if isinstance(content, dict):  # fhs
                for k, v in content.items():
                    mal_name_list.append(k)
                    mal_data.append(v)
            else:
                mal_name_list.append(os.path.splitext(os.path.basename(path))[0])
                mal_data.append(content)
        # load data: benignware
        for i, path in enumerate(self.ben_paths):
            try:
                content = pickle.load(open(path, 'rb'))
            except:
                print('cannot load data: {}'.format(path))
                continue
            ben_name_list.append(os.path.splitext(os.path.basename(path))[0])
            ben_data.append(content)

        # set label
        print('{} data: set label'.format(mode))
        if self.class_type == 'binary':  # BINARY
            mal_label = [1 for _ in mal_name_list]
            ben_label = [0 for _ in ben_name_list]
        else:
            label_dict = dict()

            if os.path.isdir(self.label_paths):  # label 파일들이 특정 폴더에 있는 경우
                for path, _, files in os.walk(self.label_paths):
                    for file in files:
                        ext = os.path.splitext(file)[-1]
                        if ext == '.csv':
                            with open(os.path.join(path, file), 'r') as f:
                                for line in csv.reader(f):
                                    md5, label = line[0], int(line[2])
                                    label_dict[md5] = label
                else:
                    mal_label = [label_dict.get(mal_name, -1) for mal_name in mal_name_list]
                    ben_label = list()
            else:
                with open(self.label_paths, 'r') as f:
                    rdr = csv.reader(f)

                    # toy dataset label
                    LABEL_TO_INT = {'Virus': 0, 'Worm': 1, 'Trojan': 2, 'not-a-virus:Downloader': 3,
                                    'Trojan-Ransom': 4, 'Backdoor': 5}
                    for line in rdr:
                        if int(line[1]) == 1:
                            md5, label = line[0], LABEL_TO_INT[line[2]]
                            label_dict[md5] = label
                    else:
                        mal_label = [label_dict.get(mal_name, -1) for mal_name in mal_name_list]
                        ben_label = list()

        # make (data:label) dictionary
        print('{} data: make data dictionary'.format(mode))
        self.mal_data_dict = dict(zip(mal_name_list, mal_data))
        self.mal_label_dict = dict(zip(mal_name_list, mal_label))
        self.ben_data_dict = dict(zip(ben_name_list, ben_data))
        self.ben_label_dict = dict(zip(ben_name_list, ben_label))

        # remove wrong data/label
        print('{} data: remove wrong data/label'.format(mode))
        md5_del_list = list()
        for mal_name in self.mal_label_dict:
            if self.mal_label_dict[mal_name] == -1:
                try:
                    del self.mal_data_dict[mal_name]
                    md5_del_list.append(mal_name)
                except:
                    pass
        else:
            for md5 in md5_del_list:
                try:
                    del self.mal_label_dict[md5]
                except:
                    pass

        del mal_name_list, mal_data, ben_name_list, ben_data, mal_label, ben_label

        mal_name_list = list(self.mal_label_dict.keys())
        ben_name_list = list(self.ben_label_dict.keys())

        self.no_mal_data = len(mal_name_list)
        self.no_ben_data = len(ben_name_list)
        self.number_of_data = self.no_mal_data + self.no_ben_data
        self.all_name_list = mal_name_list + ben_name_list

        # set batch size
        self.batch_size = batch_size

        # set epoch
        self.epoch = epoch
        pass

    def get_all_file_names(self):
        return self.all_name_list

    def __len__(self):
        return self.number_of_data

    def _mal_generator(self, batch_size):
        mal_name_list = list(self.mal_label_dict.keys())
        for epoch in range(1, self.epoch+1):
            notice = {'epoch': epoch, 'signal': False}
            random.shuffle(mal_name_list)
            mal_data = [self.mal_data_dict[mal_name] for mal_name in mal_name_list]
            mal_label = [self.mal_label_dict[mal_name] for mal_name in mal_name_list]
            for i in range(0, self.no_mal_data, batch_size):
                yield (mal_data[i:i+batch_size], mal_label[i:i+batch_size], notice)
            else:
                print('@ epoch: {}'.format(epoch))
                notice['signal'] = True
                yield (list(), list(), notice)

    def _ben_generator(self, batch_size):
        ben_name_list = list(self.ben_label_dict.keys())
        if self.class_type == 'multiclass':
            while True:
                yield (list(), list())
        else:
            while True:  # fit code to malware
                random.shuffle(ben_name_list)
                ben_data = [self.ben_data_dict[ben_name] for ben_name in ben_name_list]
                ben_label = [self.ben_label_dict[ben_name] for ben_name in ben_name_list]
                for i in range(0, self.no_ben_data, batch_size):
                    yield (ben_data[i:i+batch_size], ben_label[i:i+batch_size])

    def __iter__(self):
        if self.class_type == 'binary':
            batch_size = int(self.batch_size // 2)
        else:
            batch_size = self.batch_size

        if self.iter_mode == 'train':  # mini-batch
            for ((mal_data, mal_label, notice), (ben_data, ben_label)) in zip(self._mal_generator(batch_size),
                                                                      self._ben_generator(batch_size)):
                yield ((mal_data + ben_data), (mal_label + ben_label), notice)

        else:  # evaluation mode
            # initialize batch data/label
            mal_data = list(self.mal_data_dict.values())
            mal_label = list(self.mal_label_dict.values())
            ben_data = list(self.ben_data_dict.values())
            ben_label = list(self.ben_label_dict.values())

            total_data_list = mal_data + ben_data
            total_label_list = mal_label + ben_label
            for i in range(0, self.number_of_data, self.batch_size):
                yield (total_data_list[i:i+self.batch_size], total_label_list[i:i+self.batch_size])
