import os
import numpy as np


def read_all_npz_files(directory):
    """
    遍历目录下的所有 .npz 文件并读取其内容
    :param directory: str, 要扫描的目录路径
    """
    print(f"扫描目录: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".npz"):
                file_path = os.path.join(root, file)
                read_npz(file_path)


def read_npz(file_path):
    """
    读取并分析单个 .npz 文件的维度和内容信息
    :param file_path: str, .npz 文件路径
    """
    try:
        # 加载 .npz 文件
        data = np.load(file_path)

        print(f"\n文件: {file_path}")
        print("包含的数据键及其形状:")

        # 遍历文件中的数据
        for key in data.files:
            array = data[key]
            print(f"- {key}: {array.shape}")

        # 提供维度含义的通用说明
        print("\n键的可能含义（需根据具体数据集调整）:")
        print("- R: 原子坐标（shape: (时间步数, 原子数, 3)）")
        print("- F: 原子力（shape: (时间步数, 原子数, 3)）")
        print("- E: 能量（shape: (时间步数,)）")
        print("- Z: 原子类型（shape: (原子数,)）")
    except Exception as e:
        print(f"读取 .npz 文件时出错: {e}")


if __name__ == "__main__":
    # 指定需要扫描的目录路径
    directory_path = "md17/raw_data/md17"  # 替换为你的实际路径
    read_all_npz_files(directory_path)

# import numpy as np
#
# # 文件路径定义
# base_path = "eth_ucy/processed_data_diverse"
# datasets = ["eth", "hotel", "univ", "zara1", "zara2"]
#
#
# # 打印文件信息的函数
# def print_file_info(file_path, read_count=20, sample_count=3):
#     try:
#         # 加载数据
#         data = np.load(file_path)
#         print(f"File: {file_path}")
#         print(f"Shape: {data.shape}")  # 打印数据的维度
#
#         # 如果是 num 文件，打印前 20-30 个元素
#         if "num" in file_path:
#             print(f"First {read_count} elements (num file):")
#             print(data[:read_count])
#         # 如果是 data_train 或 data_test 文件，打印连续的 3 个样本
#         elif "data" in file_path:
#             print(f"Sample content (first {sample_count} samples):")
#             print(data[:sample_count])
#         print("-" * 50)
#     except Exception as e:
#         print(f"Error reading file {file_path}: {e}")
#
#
# # 遍历每个数据集的文件
# for dataset in datasets:
#     print(f"==== Dataset: {dataset} ====")
#
#     # 打印 num_train 和 num_test
#     num_train_path = f"{base_path}/{dataset}_num_train.npy"
#     num_test_path = f"{base_path}/{dataset}_num_test.npy"
#     print_file_info(num_train_path, read_count=30)
#     print_file_info(num_test_path, read_count=30)
#
#     # 打印 data_train 和 data_test
#     data_train_path = f"{base_path}/{dataset}_data_train.npy"
#     data_test_path = f"{base_path}/{dataset}_data_test.npy"
#     print_file_info(data_train_path, sample_count=3)
#     print_file_info(data_test_path, sample_count=3)


