import numpy as np

# 文件路径定义
base_path = "eth_ucy/processed_data_diverse"
datasets = ["eth", "hotel", "univ", "zara1_main.sh", "zara2"]


# 打印文件信息的函数
def print_file_info(file_path, read_count=20, sample_count=3):
    try:
        # 加载数据
        data = np.load(file_path)
        print(f"File: {file_path}")
        print(f"Shape: {data.shape}")  # 打印数据的维度

        # 如果是 num 文件，打印前 20-30 个元素
        if "num" in file_path:
            print(f"First {read_count} elements (num file):")
            print(data[:read_count])
        # 如果是 data_train 或 data_test 文件，打印连续的 3 个样本
        elif "data" in file_path:
            print(f"Sample content (first {sample_count} samples):")
            print(data[:sample_count])
        print("-" * 50)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")


# 遍历每个数据集的文件
for dataset in datasets:
    print(f"==== Dataset: {dataset} ====")

    # 打印 num_train 和 num_test
    num_train_path = f"{base_path}/{dataset}_num_train.npy"
    num_test_path = f"{base_path}/{dataset}_num_test.npy"
    print_file_info(num_train_path, read_count=30)
    print_file_info(num_test_path, read_count=30)

    # 打印 data_train 和 data_test
    data_train_path = f"{base_path}/{dataset}_data_train.npy"
    data_test_path = f"{base_path}/{dataset}_data_test.npy"
    print_file_info(data_train_path, sample_count=3)
    print_file_info(data_test_path, sample_count=3)


