import os

# CPU COUNT
CPU_COUNT = 10

# SCRIPT CONTROLLER OPTION
ANALYSIS_FLAG = 0
SAVE_FH_FLAG = 1

# IDB/OPS OPTION
FILE_CLASS = 'unknown'  # malware, benignware

# FH OPTION
FH_TYPE = 'fh_panr'  # counter(r), content(c), p(process), a(apics)
N_GRAM = 4
FH_CONTENT_BOUNDARY = 65536  # 2^8, 2^16, 2^32, 2^64
MAX_VECTOR_SIZE = 4096  # 4096, 2048, 1024, 512, 256, 128
MAX_VECTOR_SIZE_BIT = 12
BOUNDARY_SIZE = -1

# IDA 경로
IDA_PATH = os.path.normpath('C:/Program Files/IDA 7.0/idat64.exe')
IDA_TIME_OUT = 180

# 데이터셋 기본 경로
# BASE_PATH = os.path.normpath(os.path.abspath('D:/working_board/toy_dataset'))
BASE_PATH = os.path.normpath(os.path.abspath('D:/working_board/dataset_kisa'))

# ZIP FILE PATH
ZIP_FILE_PATH = os.path.normpath(os.path.abspath('{0}/{1}/zipfile'.format(BASE_PATH, FILE_CLASS)))

# 입력 파일 경로
INPUT_FILE_PATH = os.path.normpath(os.path.abspath('{0}/{1}/vir'.format(BASE_PATH, FILE_CLASS)))

# idb(i64) 저장 경로
IDB_PATH = os.path.normpath(os.path.abspath('{0}/{1}/idb'.format(BASE_PATH, FILE_CLASS)))

# ida python script 저장 경로
IDA_PYTHON_SCRIPT_PATH = os.path.normpath(os.path.abspath('./ida_script/ida_opcode.py'))

############################ OPS ############################
OPS_PATH = os.path.normpath(os.path.abspath('{0}/{1}/ops'.format(BASE_PATH, FILE_CLASS)))
ACS_PATH = os.path.normpath(os.path.abspath('{0}/{1}/acs'.format(BASE_PATH, FILE_CLASS)))
FH_INPUT_PATH = os.path.normpath(os.path.abspath(ACS_PATH))

############################ FH ############################
FH_PATH = os.path.normpath(os.path.abspath('{0}/{1}/{2}/{3}'.format(BASE_PATH, FILE_CLASS, FH_TYPE, MAX_VECTOR_SIZE)))


# function #
def create_file_list(root):
    ret_list = []
    for path, dirs, files in os.walk(root):
        if len(dirs) == 0:
            print('create file path: {}'.format(os.path.split(path)[-1]))
            for file in files:
                full_file_path = os.path.join(path, file)
                ret_list.append(full_file_path)
    return ret_list
