import zipfile, os
import pathos.pools as pp
import pefile

'''
- 파일 이름: VirusShare_md5
- 32bit pe file 인지 확인
'''
base_path = os.path.normpath(r'E:\virusshare')
dst_path = os.path.normpath(r'D:/working_board/virusshare/vir')
cpu_count = 5


def unzip(file_path):
    print('*'*10, file_path, 'start', '*'*10)
    with zipfile.ZipFile(file_path) as zf:
        zf.setpassword(b'infected')
        dir_name = os.path.splitext(os.path.basename(file_path))[0]

        for file in zf.namelist():
            file_name = zf.getinfo(file).filename
            md5 = file_name.replace('VirusShare_', '')
            try:
                dir_path = os.path.join(dst_path, dir_name)
                save_path = os.path.join(dir_path, md5 + '.vir')

                if os.path.exists(save_path):
                    print('{} already generated'.format(file_name))
                    continue

                data = zf.read(file)
                pe = pefile.PE(data=data)
                if pe.FILE_HEADER.Machine == 0x14C:  # Intel 32bit pe file
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    try:
                        with open(save_path, 'wb') as f:
                            f.write(data)
                    except:
                        pass
                    print('{} generated'.format(file_name))
            except Exception as e:
                # print('{} Error: {}'.format(file_name, e))
                pass
    os.remove(file_path)


def main():
    zip_path_list = [os.path.join(base_path, file) for file in os.listdir(base_path) if os.path.splitext(file)[-1] == '.zip']

    p = pp.ProcessPool(cpu_count)
    p.map(unzip, zip_path_list)


if __name__ == '__main__' :
    main()
