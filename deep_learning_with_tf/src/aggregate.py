import os, shutil

order = '4th'
extension = 'fh_bovw'
size = '2048'

base_path = r'D:\working_board\kisa_dataset\{}'.format(order)
dst_path = r'D:\working_board\kisa_dataset\{}\{}\v\{}'.format(order, extension, size)

ret = list()
for path, _, files in os.walk(base_path):
    for file in files:
        if os.path.splitext(file)[-1] == '.{}'.format(extension):
            ret.append(os.path.join(path, file))

if not os.path.exists(dst_path):
    os.makedirs(dst_path)

for each in ret:
    shutil.copy(each, os.path.join(dst_path, os.path.basename(each)))
