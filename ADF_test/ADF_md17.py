import os
import numpy as np
from statsmodels.tsa.stattools import adfuller

# 分子文件映射表
molecule_files = dict(
    Aspirin="aspirin_dft.npz",
    Benzene="benzene_dft.npz",
    Ethanol="ethanol_dft.npz",
    Malonaldehyde="malonaldehyde_dft.npz",
    Naphthalene="naphthalene_dft.npz",
    Salicylic="salicylic_dft.npz",
    Toluene="toluene_dft.npz",
    Uracil="uracil_dft.npz",
)

# 数据文件夹路径
raw_path = "../md17/raw_data/md17"

# 提取起始时间步索引设置
time_ranges = {
    'Aspirin': (1000, 211762),
    'Benzene': (1000, 627983),
    'Ethanol': (1000, 555092),
    'Malonaldehyde': (1000, 993237),
    'Naphthalene': (111000, 326250),
    'Salicylic': (11000, 320231),
    'Toluene': (12000, 442790),
    'Uracil': (11000, 133770),
}

SEGMENT_LENGTH = 500  # 每段长度
NUM_SEGMENTS = 10  # 每个分子取 10 段

def read_mol(mol_name):
    """
    读取分子数据并提取非氢原子的坐标
    """
    npz_file = np.load(os.path.join(raw_path, molecule_files[mol_name]))
    z = npz_file['z']  # 原子类型
    x_all = npz_file['R']  # 全部原子坐标
    x = x_all[:, z > 1]  # 过滤掉氢原子
    return z, x_all, x

def adf_test(series):
    """
    对单个时间序列执行 ADF 测试
    :param series: ndarray, 时间序列
    :return: dict, 包含 ADF 检验结果
    """
    result = adfuller(series)
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "critical_values": result[4],
    }

def perform_adf_test(x, molecule, start, end, file):
    """
    对特定时间片段的三维坐标进行 ADF 测试并写入文件
    """
    num_atoms = x.shape[1]
    total_p_values = []
    total_adf_statistics = []

    file.write(f"\nProcessing {molecule} - Total range: {start} to {end}\n")

    for segment_idx in range(NUM_SEGMENTS):
        segment_start = start + segment_idx * SEGMENT_LENGTH
        segment_end = segment_start + SEGMENT_LENGTH
        if segment_end > end:
            break  # 防止超出时间序列范围

        file.write(f"Segment {segment_idx + 1}: Time range {segment_start} to {segment_end}\n")

        segment_p_values = []
        segment_adf_statistics = []
        for atom_idx in range(num_atoms):
            best_result = None
            best_coord = None
            for coord in range(3):  # 分别对 x, y, z 三个方向进行测试
                result = adf_test(x[segment_start:segment_end, atom_idx, coord])
                if best_result is None or result["p-value"] > best_result["p-value"]:
                    best_result = result
                    best_coord = coord

            file.write(
                f"Molecule: {molecule}, Segment: {segment_idx + 1}, Atom: {atom_idx}, Best Coord: {best_coord}, "
                f"ADF Statistic: {best_result['ADF Statistic']:.4f}, p-value: {best_result['p-value']:.4f}, "
                f"Critical Values: 1%: {best_result['critical_values']['1%']:.3f}, "
                f"5%: {best_result['critical_values']['5%']:.3f}, "
                f"10%: {best_result['critical_values']['10%']:.3f}\n"
            )

            segment_p_values.append(best_result["p-value"])
            segment_adf_statistics.append(best_result["ADF Statistic"])

        total_p_values.extend(segment_p_values)
        total_adf_statistics.extend(segment_adf_statistics)

    return total_p_values, total_adf_statistics

def summarize_results(summary_file, molecule_stats):
    """
    汇总每个分子的统计信息
    """
    with open(summary_file, "w") as file:
        for molecule, stats in molecule_stats.items():
            file.write(f"Molecule: {molecule}\n")
            file.write(f"Total Atoms: {stats['total_atoms']}\n")
            file.write(f"Non-Stationary Atoms: {stats['non_stationary_atoms']}\n")
            file.write(f"Mean p-value: {stats['mean_p_value']:.4f}\n")
            file.write(f"Std p-value: {stats['std_p_value']:.4f}\n")
            file.write(f"Mean ADF Statistic: {stats['mean_adf_stat']:.4f}\n")
            file.write(f"Std ADF Statistic: {stats['std_adf_stat']:.4f}\n")
            file.write(f"Max ADF Statistic: {stats['max_adf_stat']:.4f}\n")
            file.write(f"Min ADF Statistic: {stats['min_adf_stat']:.4f}\n\n")

if __name__ == "__main__":
    output_file = "md17_adf_results.txt"
    summary_file = "md17_adf_summary.txt"

    # 确保输出文件为空（如果存在）
    open(output_file, "w").close()

    molecule_stats = {}

    # 遍历所有分子
    with open(output_file, "a") as file:
        for molecule, file_name in molecule_files.items():
            if molecule in time_ranges:
                try:
                    start, end = time_ranges[molecule]
                    z, x_all, x = read_mol(molecule)  # 读取数据
                    num_atoms = x.shape[1]

                    total_p_values, total_adf_statistics = perform_adf_test(x, molecule, start, end, file)

                    non_stationary_atoms = sum(p >= 0.05 for p in total_p_values)
                    molecule_stats[molecule] = {
                        "total_atoms": num_atoms,
                        "non_stationary_atoms": non_stationary_atoms,
                        "mean_p_value": np.mean(total_p_values),
                        "std_p_value": np.std(total_p_values),
                        "mean_adf_stat": np.mean(total_adf_statistics),
                        "std_adf_stat": np.std(total_adf_statistics),
                        "max_adf_stat": np.max(total_adf_statistics),
                        "min_adf_stat": np.min(total_adf_statistics),
                    }

                except Exception as e:
                    print(f"Error processing {molecule}: {e}")

    summarize_results(summary_file, molecule_stats)
    print(f"Results saved to {output_file} and {summary_file}")


# import os
# import numpy as np
# from statsmodels.tsa.stattools import adfuller
#
# # 分子文件映射表
# molecule_files = dict(
#     Aspirin="aspirin_dft.npz",
#     Benzene="benzene_dft.npz",
#     Ethanol="ethanol_dft.npz",
#     Malonaldehyde="malonaldehyde_dft.npz",
#     Naphthalene="naphthalene_dft.npz",
#     Salicylic="salicylic_dft.npz",
#     Toluene="toluene_dft.npz",
#     Uracil="uracil_dft.npz",
# )
#
# # 数据文件夹路径
# raw_path = "../md17/raw_data/md17"
#
# # 提取起始时间步索引设置
# time_ranges = {
#     'Aspirin': (1000, 211762),
#     'Benzene': (1000, 627983),
#     'Ethanol': (1000, 555092),
#     'Malonaldehyde': (1000, 993237),
#     'Naphthalene': (111000, 326250),
#     'Salicylic': (11000, 320231),
#     'Toluene': (12000, 442790),
#     'Uracil': (11000, 133770),
# }
#
# SEGMENT_LENGTH = 500  # 每段长度
# NUM_SEGMENTS = 10  # 每个分子取 10 段
#
# def read_mol(mol_name):
#     """
#     读取分子数据并提取非氢原子的坐标
#     """
#     npz_file = np.load(os.path.join(raw_path, molecule_files[mol_name]))
#     z = npz_file['z']  # 原子类型
#     x_all = npz_file['R']  # 全部原子坐标
#     x = x_all[:, z > 1]  # 过滤掉氢原子
#     return z, x_all, x
#
# def adf_test(series):
#     """
#     对单个时间序列执行 ADF 测试
#     :param series: ndarray, 时间序列
#     :return: dict, 包含 ADF 检验结果
#     """
#     result = adfuller(series)
#     return {
#         "ADF Statistic": result[0],
#         "p-value": result[1],
#         "critical_values": result[4],
#     }
#
# def perform_adf_test(x, molecule, start, end, file):
#     """
#     对特定时间片段的三维坐标进行 ADF 测试并写入文件
#     """
#     num_atoms = x.shape[1]
#     total_p_values = []
#
#     file.write(f"\nProcessing {molecule} - Total range: {start} to {end}\n")
#
#     for segment_idx in range(NUM_SEGMENTS):
#         segment_start = start + segment_idx * SEGMENT_LENGTH
#         segment_end = segment_start + SEGMENT_LENGTH
#         if segment_end > end:
#             break  # 防止超出时间序列范围
#
#         file.write(f"Segment {segment_idx + 1}: Time range {segment_start} to {segment_end}\n")
#
#         segment_p_values = []
#         for atom_idx in range(num_atoms):
#             best_result = None
#             best_coord = None
#             for coord in range(3):  # 分别对 x, y, z 三个方向进行测试
#                 result = adf_test(x[segment_start:segment_end, atom_idx, coord])
#                 if best_result is None or result["p-value"] > best_result["p-value"]:
#                     best_result = result
#                     best_coord = coord
#
#             file.write(
#                 f"Molecule: {molecule}, Segment: {segment_idx + 1}, Atom: {atom_idx}, Best Coord: {best_coord}, "
#                 f"ADF Statistic: {best_result['ADF Statistic']:.4f}, p-value: {best_result['p-value']:.4f}, "
#                 f"Critical Values: 1%: {best_result['critical_values']['1%']:.3f}, "
#                 f"5%: {best_result['critical_values']['5%']:.3f}, "
#                 f"10%: {best_result['critical_values']['10%']:.3f}\n"
#             )
#
#             segment_p_values.append(best_result["p-value"])
#
#         total_p_values.extend(segment_p_values)
#
#     return total_p_values
#
# def summarize_results(summary_file, molecule_stats):
#     """
#     汇总每个分子的统计信息
#     """
#     with open(summary_file, "w") as file:
#         for molecule, stats in molecule_stats.items():
#             file.write(f"Molecule: {molecule}\n")
#             file.write(f"Total Atoms: {stats['total_atoms']}\n")
#             file.write(f"Non-Stationary Atoms: {stats['non_stationary_atoms']}\n")
#             file.write(f"Mean p-value: {stats['mean_p_value']:.4f}\n")
#             file.write(f"Std p-value: {stats['std_p_value']:.4f}\n\n")
#
# if __name__ == "__main__":
#     output_file = "md17_adf_results.txt"
#     summary_file = "md17_adf_summary.txt"
#
#     # 确保输出文件为空（如果存在）
#     open(output_file, "w").close()
#
#     molecule_stats = {}
#
#     # 遍历所有分子
#     with open(output_file, "a") as file:
#         for molecule, file_name in molecule_files.items():
#             if molecule in time_ranges:
#                 try:
#                     start, end = time_ranges[molecule]
#                     z, x_all, x = read_mol(molecule)  # 读取数据
#                     num_atoms = x.shape[1]
#
#                     total_p_values = perform_adf_test(x, molecule, start, end, file)
#
#                     non_stationary_atoms = sum(p >= 0.05 for p in total_p_values)
#                     molecule_stats[molecule] = {
#                         "total_atoms": num_atoms,
#                         "non_stationary_atoms": non_stationary_atoms,
#                         "mean_p_value": np.mean(total_p_values),
#                         "std_p_value": np.std(total_p_values),
#                     }
#
#                 except Exception as e:
#                     print(f"Error processing {molecule}: {e}")
#
#     summarize_results(summary_file, molecule_stats)
#     print(f"Results saved to {output_file} and {summary_file}")

