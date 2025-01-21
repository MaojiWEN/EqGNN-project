import os
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings


def load_data(data_path):
    """
    加载 ETH-UCY 数据集的样例数据
    :param data_path: str, 数据文件路径
    :return: ndarray, 数据数组
    """
    data = np.load(data_path)
    print(f"Loaded data from: {data_path}")
    print(f"Data shape: {data.shape}")
    return data


def adf_test(series):
    """
    对单个时间序列执行 ADF 测试
    :param series: ndarray, 时间序列
    :return: dict, 包含 ADF 检验结果
    """
    if np.all(series == series[0]):  # 检查时间序列是否为常量
        return {"ADF Statistic": None, "p-value": None, "is_constant": True, "critical_values": None}
    if np.std(series) < 1e-3:  # 检查标准差是否接近 0
        return {"ADF Statistic": None, "p-value": None, "is_constant": True, "critical_values": None}

    # 捕获 RuntimeWarning 并处理
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error", RuntimeWarning)  # 将 RuntimeWarning 转为异常
        try:
            result = adfuller(series)
            return {
                "ADF Statistic": result[0],
                "p-value": result[1],
                "is_constant": False,
                "critical_values": result[4],
            }
        except RuntimeWarning as e:
            print(f"RuntimeWarning for series: {series[:5]}... (truncated)")
            return {"ADF Statistic": None, "p-value": None, "is_constant": True, "critical_values": None}
        except Exception as e:
            print(f"Error during ADF test: {e}")
            return {"ADF Statistic": None, "p-value": None, "is_constant": True, "critical_values": None}


def process_sample(sample, num_nodes):
    """
    对单个样例进行分析
    :param sample: ndarray, 单个样例
    :param num_nodes: int, 样例中有效节点数
    :return: list, 每个节点的最非平稳结果
    """
    node_results = []
    for node_idx in range(num_nodes):
        node_data = sample[node_idx]  # 取单个节点的数据
        best_result = None
        best_coord = None
        for coord in range(node_data.shape[1]):  # 分别对 x 和 y 方向处理
            adf_result = adf_test(node_data[:, coord])
            if adf_result["is_constant"]:
                continue  # 跳过常量序列
            # 更新最优结果：选取 p-value 最大的维度
            if best_result is None or (adf_result["p-value"] > best_result["p-value"]):
                best_result = adf_result
                best_coord = coord
        if best_result:
            node_results.append({"node_idx": node_idx, "best_coord": best_coord, **best_result})
        else:
            # 如果节点所有方向均为常量或接近常量，则计入 constant
            node_results.append(
                {"node_idx": node_idx, "best_coord": None, "ADF Statistic": None, "p-value": None, "is_constant": True,
                 "critical_values": None})
    return node_results


def process_dataset(data_path, num_path, output_file, summary_file):
    """
    对数据集执行平稳性计算
    :param data_path: str, 数据文件路径
    :param num_path: str, 节点数文件路径
    :param output_file: str, 输出结果文件
    :param summary_file: str, 每个子集的统计文件
    """
    data = load_data(data_path)
    num_data = np.load(num_path)
    total_nodes = 0
    constant_nodes = 0
    non_stationary_nodes = 0
    total_p_values = []
    total_adf_statistics = []  # 新增：记录 ADF Statistic
    max_adf_stat = -np.inf  # 用于记录最大 ADF Statistic
    min_adf_stat = np.inf  # 用于记录最小 ADF Statistic
    critical_values_1_percent = []
    critical_values_5_percent = []
    critical_values_10_percent = []

    with open(output_file, "a") as file:
        file.write(f"\nProcessing file: {data_path}\n")
        for i, sample in enumerate(data):
            num_nodes = num_data[i]  # 获取样例的节点数
            total_nodes += num_nodes
            sample_results = process_sample(sample, num_nodes)
            for result in sample_results:
                if result["is_constant"]:
                    constant_nodes += 1
                else:
                    total_p_values.append(result["p-value"])
                    total_adf_statistics.append(result["ADF Statistic"])

                    # 记录置信区间
                    critical_values_1_percent.append(result['critical_values']['1%'])
                    critical_values_5_percent.append(result['critical_values']['5%'])
                    critical_values_10_percent.append(result['critical_values']['10%'])

                    # 更新最大最小 ADF Statistic
                    if result["ADF Statistic"] is not None:
                        max_adf_stat = max(max_adf_stat, result["ADF Statistic"])
                        min_adf_stat = min(min_adf_stat, result["ADF Statistic"])

                    # 判断是否非平稳
                    if result["p-value"] > 0.05:
                        non_stationary_nodes += 1

                critical_values_str = (
                    f"1%: {result['critical_values']['1%']}, "
                    f"5%: {result['critical_values']['5%']}, "
                    f"10%: {result['critical_values']['10%']}"
                    if result["critical_values"]
                    else "N/A"
                )
                file.write(
                    f"Sample {i}, Node {result['node_idx']}, Best Coord: {result['best_coord']}, "
                    f"ADF Statistic: {result['ADF Statistic']}, p-value: {result['p-value']}, "
                    f"Critical Values: {critical_values_str}\n"
                )

    # 排序并去掉极值部分
    total_adf_statistics_sorted = sorted(total_adf_statistics)
    trim_size = int(0.05 * len(total_adf_statistics_sorted))  # 去除最大和最小的1%
    trimmed_adf_statistics = total_adf_statistics_sorted[trim_size:-trim_size]

    # 计算 ADF Statistic 的平均值
    mean_adf_stat = np.mean(trimmed_adf_statistics) if trimmed_adf_statistics else None
    std_adf_stat = np.std(trimmed_adf_statistics) if trimmed_adf_statistics else None

    # 计算置信区间的平均值
    mean_1_percent = np.mean(critical_values_1_percent) if critical_values_1_percent else None
    mean_5_percent = np.mean(critical_values_5_percent) if critical_values_5_percent else None
    mean_10_percent = np.mean(critical_values_10_percent) if critical_values_10_percent else None

    # 写入统计文件
    with open(summary_file, "a") as summary:
        summary.write(f"\nDataset: {data_path}\n")
        summary.write(f"Total Nodes: {total_nodes}\n")
        summary.write(f"Constant Nodes: {constant_nodes}\n")
        summary.write(f"Non-Stationary Nodes: {non_stationary_nodes}\n")
        summary.write(f"Max ADF Statistic: {max_adf_stat:.4f}\n")
        summary.write(f"Min ADF Statistic: {min_adf_stat:.4f}\n")
        if total_p_values:
            mean_p_value = np.mean(total_p_values)
            std_p_value = np.std(total_p_values)
            summary.write(f"Mean p-value: {mean_p_value:.4f}\n")
            summary.write(f"Std p-value: {std_p_value:.4f}\n")
        else:
            summary.write("No valid p-values computed.\n")
        if mean_adf_stat is not None:
            summary.write(f"Mean ADF Statistic: {mean_adf_stat:.4f}\n")
        if std_adf_stat is not None:
            summary.write(f"Std ADF Statistic: {std_adf_stat:.4f}\n")

        # 写入置信度平均值
        summary.write(f"Mean 1% Critical Value: {mean_1_percent:.4f}\n")
        summary.write(f"Mean 5% Critical Value: {mean_5_percent:.4f}\n")
        summary.write(f"Mean 10% Critical Value: {mean_10_percent:.4f}\n")


if __name__ == "__main__":
    # 数据路径
    base_path = "../eth_ucy/processed_data_diverse"
    dataset_files = [
        "eth_data_train.npy",
        "eth_data_test.npy",
        "hotel_data_train.npy",
        "hotel_data_test.npy",
        "univ_data_train.npy",
        "univ_data_test.npy",
        "zara1_data_train.npy",
        "zara1_data_test.npy",
        "zara2_data_train.npy",
        "zara2_data_test.npy",
    ]
    num_files = [
        "eth_num_train.npy",
        "eth_num_test.npy",
        "hotel_num_train.npy",
        "hotel_num_test.npy",
        "univ_num_train.npy",
        "univ_num_test.npy",
        "zara1_num_train.npy",
        "zara1_num_test.npy",
        "zara2_num_train.npy",
        "zara2_num_test.npy",
    ]
    # 输出文件
    overall_output_file = "eth_adf_results.txt"
    subset_summary_file = "eth_adf_summary.txt"
    # 确保输出文件为空（如果存在）
    open(overall_output_file, "w").close()
    open(subset_summary_file, "w").close()
    # 遍历数据集
    for data_file, num_file in zip(dataset_files, num_files):
        data_path = os.path.join(base_path, data_file)
        num_path = os.path.join(base_path, num_file)
        process_dataset(data_path, num_path, overall_output_file, subset_summary_file)

    print(f"Results saved to {overall_output_file} and {subset_summary_file}")

# import os
# import numpy as np
# from statsmodels.tsa.stattools import adfuller
# import warnings
#
#
# def load_data(data_path):
#     """
#     加载 ETH-UCY 数据集的样例数据
#     :param data_path: str, 数据文件路径
#     :return: ndarray, 数据数组
#     """
#     data = np.load(data_path)
#     print(f"Loaded data from: {data_path}")
#     print(f"Data shape: {data.shape}")
#     return data
#
#
# def adf_test(series):
#     """
#     对单个时间序列执行 ADF 测试
#     :param series: ndarray, 时间序列
#     :return: dict, 包含 ADF 检验结果
#     """
#     if np.all(series == series[0]):  # 检查时间序列是否为常量
#         return {"ADF Statistic": None, "p-value": None, "is_constant": True, "critical_values": None}
#     if np.std(series) < 1e-3:  # 检查标准差是否接近 0
#         return {"ADF Statistic": None, "p-value": None, "is_constant": True, "critical_values": None}
#
#     # 捕获 RuntimeWarning 并处理
#     with warnings.catch_warnings(record=True) as w:
#         warnings.simplefilter("error", RuntimeWarning)  # 将 RuntimeWarning 转为异常
#         try:
#             result = adfuller(series)
#             return {
#                 "ADF Statistic": result[0],
#                 "p-value": result[1],
#                 "is_constant": False,
#                 "critical_values": result[4],
#             }
#         except RuntimeWarning as e:
#             print(f"RuntimeWarning for series: {series[:5]}... (truncated)")
#             return {"ADF Statistic": None, "p-value": None, "is_constant": True, "critical_values": None}
#         except Exception as e:
#             print(f"Error during ADF test: {e}")
#             return {"ADF Statistic": None, "p-value": None, "is_constant": True, "critical_values": None}
#
#
# def process_sample(sample, num_nodes):
#     """
#     对单个样例进行分析
#     :param sample: ndarray, 单个样例
#     :param num_nodes: int, 样例中有效节点数
#     :return: list, 每个节点的最非平稳结果
#     """
#     node_results = []
#     for node_idx in range(num_nodes):
#         node_data = sample[node_idx]  # 取单个节点的数据
#         best_result = None
#         best_coord = None
#         for coord in range(node_data.shape[1]):  # 分别对 x 和 y 方向处理
#             adf_result = adf_test(node_data[:, coord])
#             if adf_result["is_constant"]:
#                 continue  # 跳过常量序列
#             # 更新最优结果：选取 p-value 最大的维度
#             if best_result is None or (adf_result["p-value"] > best_result["p-value"]):
#                 best_result = adf_result
#                 best_coord = coord
#         if best_result:
#             node_results.append({"node_idx": node_idx, "best_coord": best_coord, **best_result})
#         else:
#             # 如果节点所有方向均为常量或接近常量，则计入 constant
#             node_results.append(
#                 {"node_idx": node_idx, "best_coord": None, "ADF Statistic": None, "p-value": None, "is_constant": True,
#                  "critical_values": None})
#     return node_results
#
#
# def process_dataset(data_path, num_path, output_file, summary_file):
#     """
#     对数据集执行平稳性计算
#     :param data_path: str, 数据文件路径
#     :param num_path: str, 节点数文件路径
#     :param output_file: str, 输出结果文件
#     :param summary_file: str, 每个子集的统计文件
#     """
#     data = load_data(data_path)
#     num_data = np.load(num_path)
#     total_nodes = 0
#     constant_nodes = 0
#     non_stationary_nodes = 0
#     total_p_values = []
#     total_adf_statistics = []  # 新增：记录 ADF Statistic
#     max_adf_stat = -np.inf  # 用于记录最大 ADF Statistic
#     min_adf_stat = np.inf  # 用于记录最小 ADF Statistic
#
#     with open(output_file, "a") as file:
#         file.write(f"\nProcessing file: {data_path}\n")
#         for i, sample in enumerate(data):
#             num_nodes = num_data[i]  # 获取样例的节点数
#             total_nodes += num_nodes
#             sample_results = process_sample(sample, num_nodes)
#             for result in sample_results:
#                 if result["is_constant"]:
#                     constant_nodes += 1
#                 else:
#                     total_p_values.append(result["p-value"])
#                     total_adf_statistics.append(result["ADF Statistic"])
#
#                     # 更新最大最小 ADF Statistic
#                     if result["ADF Statistic"] is not None:
#                         max_adf_stat = max(max_adf_stat, result["ADF Statistic"])
#                         min_adf_stat = min(min_adf_stat, result["ADF Statistic"])
#
#                     # 判断是否非平稳
#                     if result["p-value"] > 0.05:
#                         non_stationary_nodes += 1
#
#                 critical_values_str = (
#                     f"1%: {result['critical_values']['1%']}, "
#                     f"5%: {result['critical_values']['5%']}, "
#                     f"10%: {result['critical_values']['10%']}"
#                     if result["critical_values"]
#                     else "N/A"
#                 )
#                 file.write(
#                     f"Sample {i}, Node {result['node_idx']}, Best Coord: {result['best_coord']}, "
#                     f"ADF Statistic: {result['ADF Statistic']}, p-value: {result['p-value']}, "
#                     f"Critical Values: {critical_values_str}\n"
#                 )
#
#     # 排序并去掉极值部分
#     total_adf_statistics_sorted = sorted(total_adf_statistics)
#     trim_size = int(0.01 * len(total_adf_statistics_sorted))  # 去除最大和最小的3%
#     trimmed_adf_statistics = total_adf_statistics_sorted[trim_size:-trim_size]
#
#     # 计算 ADF Statistic 的平均值
#     mean_adf_stat = np.mean(trimmed_adf_statistics) if trimmed_adf_statistics else None
#     std_adf_stat = np.std(trimmed_adf_statistics) if trimmed_adf_statistics else None
#
#     # 写入统计文件
#     with open(summary_file, "a") as summary:
#         summary.write(f"\nDataset: {data_path}\n")
#         summary.write(f"Total Nodes: {total_nodes}\n")
#         summary.write(f"Constant Nodes: {constant_nodes}\n")
#         summary.write(f"Non-Stationary Nodes: {non_stationary_nodes}\n")
#         summary.write(f"Max ADF Statistic: {max_adf_stat:.4f}\n")
#         summary.write(f"Min ADF Statistic: {min_adf_stat:.4f}\n")
#         if total_p_values:
#             mean_p_value = np.mean(total_p_values)
#             std_p_value = np.std(total_p_values)
#             summary.write(f"Mean p-value: {mean_p_value:.4f}\n")
#             summary.write(f"Std p-value: {std_p_value:.4f}\n")
#         else:
#             summary.write("No valid p-values computed.\n")
#         if mean_adf_stat is not None:
#             summary.write(f"Mean ADF Statistic: {mean_adf_stat:.4f}\n")
#         if std_adf_stat is not None:
#             summary.write(f"Std ADF Statistic: {std_adf_stat:.4f}\n")
#
#
# if __name__ == "__main__":
#     # 数据路径
#     base_path = "../eth_ucy/processed_data_diverse"
#     dataset_files = [
#         "eth_data_train.npy",
#         "eth_data_test.npy",
#         "hotel_data_train.npy",
#         "hotel_data_test.npy",
#         "univ_data_train.npy",
#         "univ_data_test.npy",
#         "zara1_data_train.npy",
#         "zara1_data_test.npy",
#         "zara2_data_train.npy",
#         "zara2_data_test.npy",
#     ]
#     num_files = [
#         "eth_num_train.npy",
#         "eth_num_test.npy",
#         "hotel_num_train.npy",
#         "hotel_num_test.npy",
#         "univ_num_train.npy",
#         "univ_num_test.npy",
#         "zara1_num_train.npy",
#         "zara1_num_test.npy",
#         "zara2_num_train.npy",
#         "zara2_num_test.npy",
#     ]
#     # 输出文件
#     overall_output_file = "eth_adf_results.txt"
#     subset_summary_file = "eth_adf_summary.txt"
#     # 确保输出文件为空（如果存在）
#     open(overall_output_file, "w").close()
#     open(subset_summary_file, "w").close()
#     # 遍历数据集
#     for data_file, num_file in zip(dataset_files, num_files):
#         data_path = os.path.join(base_path, data_file)
#         num_path = os.path.join(base_path, num_file)
#         process_dataset(data_path, num_path, overall_output_file, subset_summary_file)
#
#     print(f"Results saved to {overall_output_file} and {subset_summary_file}")