import numpy as np
from sklearn.covariance import EmpiricalCovariance


def calculate_covariance_matrix(feature_paths, save_path):
    all_features = []

    # 加载所有特征数据
    for path in feature_paths:
        features = np.load(path)  # 形状应为 (N_samples, D_feature)

        # 如果特征是一维的，转换为二维
        if features.ndim == 1:
            features = features.reshape(1, -1)  # 转换为 (1, D_feature)

        all_features.append(features)

    # 合并特征
    X = np.concatenate(all_features, axis=0)
    print(f"合并后特征数据形状: {X.shape}")  # 应为 (N_total_samples, D_feature)

    # 计算协方差矩阵
    cov_estimator = EmpiricalCovariance().fit(X)
    cov_matrix = cov_estimator.covariance_

    # 保存结果
    np.save(save_path, cov_matrix)
    print(f"协方差矩阵已保存至 {save_path}，形状为 {cov_matrix.shape}")





# 使用示例
if __name__ == "__main__":
    calculate_covariance_matrix(
        feature_paths=["../../data/1/0225+505015.npy", "../../data/111/0225+352006.npy"],
        save_path="../../data/global_cov.npy"
    )
