import os

def process_list(list_file, output_file):
    with open(list_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f_out:
        for line in lines:
            parts = line.strip().split('/')
            path_label = parts[-1]
            f_out.write(f"{path_label}\n")

if __name__ == "__main__":
    list_file1 = '/home/linrui/moe/TimeMoE/lists/k600_test/k160_test_split1.txt'
    list_file2 = '/home/linrui/moe/TimeMoE/lists/k600_test/k160_test_split2.txt'
    list_file3 = '/home/linrui/moe/TimeMoE/lists/k600_test/k160_test_split3.txt'
    output_file1 = '/home/linrui/moe/TimeMoE/lists/k600_test/k160_testlist_split1.txt'
    output_file2 = '/home/linrui/moe/TimeMoE/lists/k600_test/k160_testlist_split2.txt'
    output_file3 = '/home/linrui/moe/TimeMoE/lists/k600_test/k160_testlist_split3.txt'
    process_list(list_file1, output_file1)
    process_list(list_file2, output_file2)
    process_list(list_file3, output_file3)
