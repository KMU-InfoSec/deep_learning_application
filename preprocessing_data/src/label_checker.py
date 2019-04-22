import simplejson
import pathos.pools as pp

from settings import *
from peheader_parser import parse


def move(file_path):
    dir_path = os.path.dirname(file_path).split(os.sep)
    dir_path[-2] = 'vir_no_kas'
    dir_path = '{}'.format(os.sep).join(dir_path)

    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except:
            pass

    os.rename(file_path, os.path.join(dir_path, os.path.basename(file_path)))
    pass


def check(file_path):
    dir_path = os.path.dirname(file_path).split(os.sep)
    dir_path[-2] = 'vt_report'

    report_path = os.path.join('{}'.format(os.sep).join(dir_path), os.path.basename(file_path).replace('.vir', '.json'))

    try:
        with open(report_path, 'r') as f:
            report = simplejson.load(f)

        if not report['scans']['Kaspersky']['detected']:
            move(file_path)
    except:
        move(file_path)
        pass
    pass


def main():
    file_list = create_file_list(INPUT_FILE_PATH)

    p = pp.ProcessPool(CPU_COUNT)
    p.map(check, file_list)

    # pe header parsing
    file_list = create_file_list(INPUT_FILE_PATH)
    p.map(parse, file_list)


if __name__ == '__main__':
    main()
