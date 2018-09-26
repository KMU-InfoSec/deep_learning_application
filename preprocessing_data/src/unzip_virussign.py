# 국민대학교 정보보호 연구실을 위해 존재하는 파일

import zipfile, os

from preprocessing.settings import *

import multiprocessing as mp


def create_zip_file_path( root ) :
    ret_list = []
    for path, dirs, files in os.walk(root) :
        for file in files :
            ext = os.path.splitext(file)[-1]
            if ext == '.zip' :
                full_path = os.path.join(path, file)
                ret_list.append(full_path)

    return ret_list


def unzip( file_path ) :
    zf = zipfile.ZipFile(file_path)
    date = os.path.basename(file_path).split('_')[1]
    for file in zf.namelist():
        file_name = zf.getinfo(file).filename
        if ('exe32' in file_name or 'dll32' in file_name ) and '.vir' in file_name:
            dst_path = os.path.join(INPUT_FILE_PATH + os.sep + date, os.path.basename(file_name)).replace('virussign.com_','')
            dir_path = os.path.split(dst_path)[0]
            if not os.path.exists(dir_path) :
                os.makedirs(dir_path)
            try :
                with open(dst_path, 'wb') as f :
                    f.write(zf.read(file))
            except :
                pass
    zf.close()
    os.remove(file_path)


def run() :
    mp.freeze_support()
    zip_file_path = create_zip_file_path(ZIP_FILE_PATH)
    p=mp.Pool(CPU_COUNT)
    p.map(unzip, zip_file_path)


if __name__ == '__main__' :
    run()