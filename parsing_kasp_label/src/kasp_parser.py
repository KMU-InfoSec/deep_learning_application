import json
import os
import pickle
import time

base_path = os.path.normpath('REPORT_PATH')


def parse():
    start_time = time.time()

    cnt = 0
    kasp_group_dict = dict()
    for path, dirs, files in os.walk(base_path):
        for file in files:
            cnt += 1
            src_path = os.path.join(path, file)

            # check json type
            if not os.path.splitext(file)[-1] == '.json':
                continue

            # load json file
            with open(src_path, 'r') as f:
                json_data = json.load(f)

            # read kaspersky info
            try:
                kasp_info = json_data['scans']['Kaspersky']
            except:
                print('@ {0} no have Kaspersky Label!'.format(file))
                continue

            # parse kasp info
            if kasp_info['detected'] == True:
                group_name = kasp_info['result']
                kasp_group_dict[group_name] = kasp_group_dict.get(group_name, 0) + 1

            # semi-check
            if cnt % 10000 == 0:
                print('{0}/{1}, time: {2}'.format(cnt, 6047841, time.time() - start_time))
                # save result file
                with open('kasp_group.pickle', 'wb') as f:
                    pickle.dump(kasp_group_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # check
    print('# of group: {}'.format(len(kasp_group_dict)))

    # save result file
    with open('kasp_group.pickle', 'wb') as f:
        pickle.dump(kasp_group_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    pass