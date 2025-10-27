import pandas as pd
import numpy as np

def analyze_excel(file_path: str) -> str:
    """
    自动分析Excel数据集的结构与特征，生成自然语言描述。
    """
    df = pd.read_csv(file_path)
    n_rows, n_cols = df.shape
    description = []

    # 基本信息
    description.append(f"该数据集共有 {n_rows} 条样本，包含 {n_cols} 个特征。")

    # 判断目标列
    possible_targets = [c for c in df.columns if c.lower() in ["target", "label", "y", "output", "result"]]
    if possible_targets:
        target_col = possible_targets[0]
        description.append(f"检测到目标列为 '{target_col}'。")
        target = df[target_col]

        # 判断任务类型
        if target.dtype in [np.float64, np.int64]:
            if target.nunique() < n_rows * 0.05:  # 少量唯一值，分类
                task_type = "分类"
            else:
                task_type = "回归"
        else:
            task_type = "分类"
        description.append(f"任务类型判断为 {task_type}。")
    else:
        task_type = "未知"
        description.append("未检测到明确的目标列，可能为无监督学习任务。")

    # 特征类型统计
    num_features = df.select_dtypes(include=[np.number]).shape[1]
    cat_features = df.select_dtypes(exclude=[np.number]).shape[1]
    description.append(f"数值型特征 {num_features} 个，类别型特征 {cat_features} 个。")

    # 缺失值情况
    missing_ratio = df.isnull().mean().mean()
    if missing_ratio > 0:
        description.append(f"整体缺失值比例约为 {missing_ratio:.2%}。")
    else:
        description.append("数据中无缺失值。")

    # 时间特征判断
    time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if time_cols:
        description.append(f"检测到时间序列特征列：{', '.join(time_cols)}，该任务可能涉及时序预测。")

    # 汇总描述
    summary = " ".join(description)
    return summary

# if __name__ == "__main__":
#     file_path = "./lang_rag/data/environment_data_export_2025-10-16_164127.csv"  # 替换为实际文件路径
#     summary = analyze_excel(file_path)
#     print(summary)