import csv
import numpy as np
import os
import _pickle
import random
import queue as q


class DataLoader:
    def __init__(self, mal_paths, ben_paths, label_paths, batch_size, mode):
        self.class_description = 'PE File Data Loader'
        self.iter_mode = mode  # for mini-batch data feeding
        print('{}: {} mode'.format(self.class_description, mode))

        # initialize member variable
        self.mal_paths = mal_paths
        self.ben_paths = ben_paths
        self.file_paths = np.concatenate((self.mal_paths, self.ben_paths), axis=0)
        self.label_paths = label_paths
        self.mal_data = q.Queue()
        self.ben_data = q.Queue()

        # allocate all data into memory
        print('data: set data into memory')
        _cnt = 0
        for path in self.mal_paths:
            content = _pickle.load(open(path, 'rb'))
            print('read')
            if isinstance(content[0], list):  # fhs
                for l in content:
                    _cnt += 1
                    self.mal_data.put(l)
            else:
                _cnt += 1
                self.mal_data.put(content)
            if _cnt % 10000 == 0:
                print('on reading malware: {}'.format(_cnt))
        for path in self.ben_paths:
            content = _pickle.load(open(path, 'rb'))
            self.ben_data.put(content)
        self.mal_data = list(self.mal_data.queue)
        self.ben_data = list(self.ben_data.queue)

        self.number_of_data = len(self.mal_data) + len(self.ben_data)

        # set label data
        print('data: set label')
        if len(self.ben_paths) != 0:  # BINARY
            self.mal_label = [1 for _ in self.mal_data]
            self.ben_label = [0 for _ in self.ben_data]
        else:
            self.label_dict = dict()
            with open(self.label_paths, 'r') as f:
                rdr = csv.reader(f)

                # toy dataset label
                LABEL_TO_INT = {'Virus': 0, 'Worm': 1, 'Trojan': 2, 'not-a-virus:Downloader': 3,
                                'Trojan-Ransom': 4, 'Backdoor': 5}
                for line in rdr:
                    if int(line[1]) == 1:
                        md5, label = line[0], LABEL_TO_INT[line[2]]
                        self.label_dict[md5] = label
                    else:
                        continue

                self.mal_label = [self.label_dict[os.path.splitext(os.path.split(mal_path)[-1])[0]]
                                  for mal_path in self.mal_paths]
                self.ben_label = list()
                pass

        # set batch size
        self.batch_size = batch_size
        pass

    def get_all_file_paths(self):
        return self.file_paths

    def get_batch(self):
        return self.batch_size

    def __len__(self):
        return self.number_of_data

    def __iter__(self):
        half_batch_size = int(self.batch_size // 2)
        if self.iter_mode == 'train':  # mini-batch
            while True:
                '''
                    initialize batch data/label
                '''
                batch_data_lists, batch_label_lists = list(), list()

                '''
                    shuffle index list
                '''
                mal_idx_lists = np.arange(len(self.mal_data))
                ben_idx_lists = np.arange(len(self.ben_data))

                random.shuffle(mal_idx_lists)
                random.shuffle(ben_idx_lists)

                '''
                    create batch file/label list
                '''

                if len(self.ben_paths) != 0:  # BINARY
                    for mal_idx, ben_idx in zip(mal_idx_lists[:half_batch_size], ben_idx_lists[:half_batch_size]):
                        # batch malware data
                        batch_data_lists.append(self.mal_data[mal_idx])
                        batch_label_lists.append(self.mal_label[mal_idx])
                        # batch benignware data
                        batch_data_lists.append(self.ben_data[ben_idx])
                        batch_label_lists.append(self.ben_label[ben_idx])
                else:
                    for mal_idx in mal_idx_lists[:self.batch_size]:
                        batch_data_lists.append(self.mal_data[mal_idx])
                        batch_label_lists.append(self.mal_label[mal_idx])

                yield (batch_data_lists, batch_label_lists)
        else:  # evaluation mode
            '''
                initialize batch data/label
            '''
            batch_data_lists, batch_label_lists = list(), list()

            if len(self.ben_paths) != 0:  # BINARY
                for idx, (data, label) in enumerate(zip(np.concatenate((self.mal_data, self.ben_data), axis=0),
                                                        np.concatenate((self.mal_label, self.ben_label), axis=0))):
                    if idx % self.batch_size == 0 and idx != 0:
                        yield batch_data_lists, batch_label_lists
                        batch_data_lists.clear()
                        batch_label_lists.clear()

                    batch_data_lists.append(data)
                    batch_label_lists.append(label)
                else:
                    yield batch_data_lists, batch_label_lists
            else:
                for idx, (data, label) in enumerate(zip(self.mal_data, self.mal_label)):
                    if idx % self.batch_size == 0 and idx != 0:
                        yield batch_data_lists, batch_label_lists
                        batch_data_lists.clear()
                        batch_label_lists.clear()

                    batch_data_lists.append(data)
                    batch_label_lists.append(label)
                else:
                    yield batch_data_lists, batch_label_lists
