import pickle

MAX_DEPTH = 0


def write_to_file(filename, buffer):
    print('write to file')
    with open(filename, 'w') as f:
        for line in buffer:
            f.write('{0}\n'.format(line))


class TreeNode:
    def __init__(self, label, cnt):
        # initialize
        self.name = str()
        self.num_of_files = 0
        self.child_list = list()

        self.add_node(label, cnt)

    def get_name(self):
        return self.name

    def get_num_of_files(self):
        return self.num_of_files

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        # if have no child
        if len(self.child_list) == 0:
            return self.num_of_files
        else:
            if self.index >= len(self.child_list):
                raise StopIteration

            node = self.child_list[self.index]
            self.index += 1

            return node

    def add_node(self, label, cnt):
        self.name = label.split('.')[0]
        self.num_of_files += cnt

        splited_label = label.split('.')[1:]
        if len(splited_label) == 0:
            pass
        else:
            for child in self.child_list:
                if splited_label[0] == child.get_name():
                    child.add_node('.'.join(splited_label), cnt)
                    break
            else:
                self.child_list.append(TreeNode('.'.join(splited_label), cnt))
        pass

    def _has_child(self):
        if len(self.child_list) > 0:
            return True
        else:
            return False

    def pre_order(self, depth, depth_boundary, buffer, group_list):
        if depth >= depth_boundary + 1:
            return

        log = '{0} {1}\t\t{2}'.format(depth * '-----------------', self.name, self.num_of_files)
        buffer.append(log)
        for child in self.child_list:
            child.pre_order(depth + 1, depth_boundary, buffer, group_list)
        pass


class TreeRoot:
    def __init__(self, child_list=list()):
        self.child_list = list()
        self.traversal_log = list()

        if len(child_list) > 0:
            self.child_list = child_list

    def get_child_list(self):
        return self.child_list

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.child_list):
            raise StopIteration

        node = self.child_list[self.index]
        self.index += 1

        return node

    def add_node(self, label, cnt):
        behavior = label.split('.')[0]
        for child in self.child_list:
            if behavior == child.get_name():
                child.add_node(label, cnt)
                break
        else:
            self.child_list.append(TreeNode(label, cnt))
            pass

    def traversal(self, write_flag=False, group_list=list()):
        print('@ traversal')

        for child in self.child_list:
            child.pre_order(0, MAX_DEPTH, self.traversal_log, group_list)

        if write_flag:
            write_to_file('log.txt', self.traversal_log)


def make_label_tree(child_file_name=None):
    print('@ create label tree')

    if child_file_name is None:
        # initialize tree
        root = TreeRoot()

        # read file
        with open('kasp_label.dict', 'rb') as f:
            kasp_group_dict = pickle.load(f)

        total_no_labels = len(kasp_group_dict)
        print('total label #: {}'.format(total_no_labels))

        # split label & put into label tree
        for i, (label, cnt) in enumerate(kasp_group_dict.items()):
            if i != 0 and i % 10000 == 0:
                print('{0}/{1} node added'.format(i, total_no_labels))
            root.add_node(label.lower(), cnt)
        else:
            with open('tree_child.list', 'wb') as f:
                pickle.dump(root.get_child_list(), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        root = TreeRoot(pickle.load(open(child_file_name, 'rb')))

    return root


def run(root):
    group_list = list()

    for behavior in root:
        name, no_files = behavior.get_name(), behavior.get_num_of_files()
        print(type(behavior), name, no_files)
        for operation in behavior:
            print(operation)
            for detailed in operation:
                print(detailed)
                for genes in detailed:
                    break
                break
            break
        break
