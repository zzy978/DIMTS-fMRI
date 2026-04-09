import numpy as np

pred = np.load('/data/home/zyzeng/project/DiMTS/OUTPUT/fmri_seq256_pred128_ood/ddpm_predict_fmri_seq256_pred128_ood.npy')
gt = np.load('/data/home/zyzeng/project/DiMTS/OUTPUT/fmri_seq256_pred128_ood/samples/fmri_ground_truth_256_test.npy')

pred_len = 128   # 改成你实际预测长度

pred_future = pred[:, -pred_len:, :]
gt_future = gt[:, -pred_len:, :]

err = pred_future - gt_future

mae = np.mean(np.abs(err))
mse = np.mean(err ** 2)
rmse = np.sqrt(mse)

# 避免除零
eps = 1e-8
mape = np.mean(np.abs(err) / (np.abs(gt_future) + eps)) * 100

print('pred_future shape:', pred_future.shape)
print('gt_future shape:', gt_future.shape)
print(f'MAE  = {mae:.6f}')
print(f'MSE  = {mse:.6f}')
print(f'RMSE = {rmse:.6f}')
print(f'MAPE = {mape:.4f}%')

print("-"*30)

mae_per_dim = np.mean(np.abs(err), axis=(0, 1))
rmse_per_dim = np.sqrt(np.mean(err ** 2, axis=(0, 1)))

# print('MAE per dim:', mae_per_dim)
# print('RMSE per dim:', rmse_per_dim)

import matplotlib.pyplot as plt

sample_id = 100
feature_id = 200

plt.plot(gt[sample_id, :, feature_id], label='Ground Truth')
plt.plot(pred[sample_id, :, feature_id], label='Prediction')
plt.axvline(gt.shape[1] - pred_len - 1, color='r', linestyle='--', label='Prediction Start')
plt.legend()
# plt.show()
plt.savefig('111fmri_pred256_subjects_prediction_vs_gt_minmax_sample100.png')
