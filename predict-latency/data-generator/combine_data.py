import os


def delete(file_list: list):
    # delete previous files.
    for file_name in file_list:
        if os.path.exists(file_name):
            os.remove(file_name)


def combine(file_list: list, result_name: str):
    if os.path.isfile(result_name) is True:
        return print('result_name already exists')
    else:
        out = open(result_name, 'w')
        cfg_seen = set()
        duplicate_count = 0
        line_count = 0

        # combine and check duplicates
        for file_name in file_list:
            for line in open(file_name, 'r'):
                split = line.split(',')
                if split[0] not in cfg_seen:
                    out.write(line)
                    cfg_seen.add(split[0])
                    line_count += 1
                else:
                    duplicate_count += 1

        print('total line is: ', line_count)
        print('duplicated data is total: ', duplicate_count)

        out.close()


if __name__ == '__main__':
    """
        which files is combined -> filename_list
    """
    filename_list = ['training-data/training_data_00',
                     'training-data/training_data_01',
                     'training-data/training_data_02',
                     'training-data/training_data_03',
                     'training-data/training_data_04',
                     'training-data/training_data_05',
                     'combined_total_data_2',
                     # 'training_data_1',
                     # 'training_data_2'
                     ]

    combine(filename_list, result_name="combined_total_data_3")

    """
        Warning: Delete
    """
    # delete(filename_list)
