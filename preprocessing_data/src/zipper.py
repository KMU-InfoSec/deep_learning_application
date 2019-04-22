import multiprocessing as mp
import time
import zipfile

from settings import *

ZIP_PATH = r'D:\working_board\dataset\unused\malware\fh_bovc\1024'
SAVE_PATH = ZIP_PATH


def create_dir_list(file_path):
    dir_list = list()
    for path in os.listdir(file_path):
        full_path = os.path.join(file_path, path)
        if os.path.isdir(full_path):
            dir_list.append(full_path)

    return dir_list


def zip_files(dir_path, ext):
    start_time = time.time()
    dir_name = os.path.split(dir_path)[-1]
    print('{} start'.format(dir_name))

    zip_pointer = zipfile.ZipFile(os.path.join(ZIP_PATH, '{}.zip'.format(dir_name)), 'w')

    for path, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.splitext(file)[-1] == ext:
                zip_pointer.write(os.path.join(path, file), os.path.relpath(os.path.join(path, file), dir_path),
                                  compress_type=zipfile.ZIP_DEFLATED)
    zip_pointer.close()

    print('{0} finished: {1}'.format(dir_name, time.time() - start_time))
    pass


def create_zip_file_path(root):
    ret_list = []
    for path, dirs, files in os.walk(root):
        if len(dirs) == 0:
            print(path)
            for file in files:
                ext = os.path.splitext(file)[-1]
                if ext == '.zip':
                    full_path = os.path.join(path, file)
                    ret_list.append(full_path)

    return ret_list


def unzip_files(file_path):
    zf = zipfile.ZipFile(file_path)
    date = os.path.splitext(os.path.basename(file_path))[0]
    print('{} unzip start'.format(date))
    for file in zf.namelist():
        file_name = os.path.basename(zf.getinfo(file).filename)

        dst_path = os.path.join(SAVE_PATH + os.sep + date, os.path.basename(file_name))
        dir_path = os.path.split(dst_path)[0]

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        try:
            with open(dst_path, 'wb') as f:
                f.write(zf.read(file))
        except:
            pass
    zf.close()
    os.remove(file_path)
    print('{} unzip finish'.format(date))


if __name__ == '__main__':
    mp.freeze_support()
    p = mp.Pool(CPU_COUNT)

    base_path = ZIP_PATH

    input_dir_list = create_zip_file_path(base_path)

    # unzip_files(os.path.join(ZIP_PATH, '20170802.zip'))
    p.map(unzip_files, input_dir_list)

    pass
