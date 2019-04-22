# GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007
#
# Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

from collections import OrderedDict
import pathos.pools as pp
import pefile
import pickle

from settings import *

DOS_HEADER = OrderedDict([
    ("e_magic", 0),
    ("e_cblp", 0),
    ("e_cp", 0),
    ("e_crlc", 0),
    ("e_cparhdr", 0),
    ("e_minalloc", 0),
    ("e_maxalloc", 0),
    ("e_ss", 0),
    ("e_sp", 0),
    ("e_csum", 0),
    ("e_ip", 0),
    ("e_cs", 0),
    ("e_lfarlc", 0),
    ("e_ovno", 0),
    ("e_res", bytes([0] * 8)),
    ("e_oemid", 0),
    ("e_oeminfo", 0),
    ("e_res2", bytes([0] * 20)),
    ("e_lfanew", 0)
])

FILE_HEADER = OrderedDict([
    ('Machine', 0),
    ('NumberOfSections', 0),
    ('TimeDateStamp', 0),
    ('PointerToSymbolTable', 0),
    ('NumberOfSymbols', 0),
    ('SizeOfOptionalHeader', 0),
    ('Characteristics', 0)
])

OPTIONAL_HEADER = OrderedDict([
    ('Magic', 0),
    ('MajorLinkerVersion', 0),
    ('MinorLinkerVersion', 0),
    ('SizeOfCode', 0),
    ('SizeOfInitializedData', 0),
    ('SizeOfUninitializedData', 0),
    ('AddressOfEntryPoint', 0),
    ('BaseOfCode', 0),
    ('BaseOfData', 0),
    ('ImageBase', 0),
    ('SectionAlignment', 0),
    ('FileAlignment', 0),
    ('MajorOperatingSystemVersion', 0),
    ('MinorOperatingSystemVersion', 0),
    ('MajorImageVersion', 0),
    ('MinorImageVersion', 0),
    ('MajorSubsystemVersion', 0),
    ('MinorSubsystemVersion', 0),
    ('Reserved1', 0),
    ('SizeOfImage', 0),
    ('SizeOfHeaders', 0),
    ('CheckSum', 0),
    ('Subsystem', 0),
    ('DllCharacteristics', 0),
    ('SizeOfStackReserve', 0),
    ('SizeOfStackCommit', 0),
    ('SizeOfHeapReserve', 0),
    ('SizeOfHeapCommit', 0),
    ('LoaderFlags', 0),
    ('NumberOfRvaAndSizes', 0)
])

DATA_DIRECTORY = OrderedDict([
    ('IMAGE_DIRECTORY_ENTRY_EXPORT', 0),
    ('IMAGE_DIRECTORY_ENTRY_IMPORT', 0),
    ('IMAGE_DIRECTORY_ENTRY_RESOURCE', 0),
    ('IMAGE_DIRECTORY_ENTRY_EXCEPTION', 0),
    ('IMAGE_DIRECTORY_ENTRY_SECURITY', 0),
    ('IMAGE_DIRECTORY_ENTRY_BASERELOC', 0),
    ('IMAGE_DIRECTORY_ENTRY_DEBUG', 0),
    ('IMAGE_DIRECTORY_ENTRY_COPYRIGHT', 0),
    ('IMAGE_DIRECTORY_ENTRY_GLOBALPTR', 0),
    ('IMAGE_DIRECTORY_ENTRY_TLS', 0),
    ('IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG', 0),
    ('IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT', 0),
    ('IMAGE_DIRECTORY_ENTRY_IAT', 0),
    ('IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT', 0),
    ('IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR', 0),
    ('IMAGE_DIRECTORY_ENTRY_RESERVED', 0)
])


def get_pe_header_report(file_path):
    ret = list()
    # ret = OrderedDict()
    # ret['FILE_HEADER'] = OrderedDict()
    # ret['OPTIONAL_HEADER'] = OrderedDict()
    # ret['DATA_DIRECTORY'] = OrderedDict()
    # ret['DOS_HEADER'] = OrderedDict()

    try:
        pe = pefile.PE(file_path)
    except:
        return ret

    if hasattr(pe, 'FILE_HEADER'):
        for each in FILE_HEADER:
            ret.append(getattr(pe.FILE_HEADER, each, FILE_HEADER[each]))
            # ret['FILE_HEADER'][each] = getattr(pe.FILE_HEADER, each, FILE_HEADER[each])

    if hasattr(pe, 'OPTIONAL_HEADER'):
        for each in OPTIONAL_HEADER:
            ret.append(getattr(pe.OPTIONAL_HEADER, each, OPTIONAL_HEADER[each]))
            # ret['OPTIONAL_HEADER'][each] = getattr(pe.OPTIONAL_HEADER, each, OPTIONAL_HEADER[each])

        if hasattr(pe.OPTIONAL_HEADER, 'DATA_DIRECTORY'):
            for each in DATA_DIRECTORY:
                ret.append(getattr(pe.OPTIONAL_HEADER.DATA_DIRECTORY, each, DATA_DIRECTORY[each]))
                # ret['DATA_DIRECTORY'][each] = getattr(pe.OPTIONAL_HEADER.DATA_DIRECTORY, each, DATA_DIRECTORY[each])
        else:
            ret.extend([0] * len(DATA_DIRECTORY))

    if hasattr(pe, 'DOS_HEADER'):
        for each in DOS_HEADER:
            value = getattr(pe.DOS_HEADER, each, DOS_HEADER[each])
            if isinstance(value, bytes):
                ret.append(int.from_bytes(value, byteorder='little'))
            else:
                ret.append(value)
            # ret['DOS_HEADER'][each] = getattr(pe.DOS_HEADER, each, DOS_HEADER[each])

    return ret


def parse(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    sub_file_path = file_path.replace(INPUT_FILE_PATH, '').replace(os.path.basename(file_path), '')

    save_path = PEHDR_PATH + sub_file_path
    dst_path = os.path.join(save_path, file_name) + '.pehdr'

    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    ret = get_pe_header_report(file_path)
    with open(dst_path, 'wb') as f:
        pickle.dump(ret, f)
    pass


def main():
    file_list = create_file_list(INPUT_FILE_PATH)

    p = pp.ProcessPool(CPU_COUNT)
    p.map(parse, file_list)
    pass


if __name__ == '__main__':
    main()
