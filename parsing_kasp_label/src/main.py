from tree_maker import *

if __name__ == '__main__':
    root = make_label_tree(child_file_name='tree_child.list')

    group_list = list()
    root.traversal(group_list=group_list)

    run(root)
