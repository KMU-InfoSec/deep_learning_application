from tree_maker import *

if __name__ == '__main__':
    with open('kasp_label.dict', 'rb') as f:
        content = pickle.load(f)

    print(len(content))
    input()

    print(content)

    input()

    root = make_label_tree(child_file_name='tree_child.list')

    group_list = list()
    root.traversal(group_list=group_list)

    # run(root)
