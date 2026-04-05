import numpy as np

a = np.load("/data/home/zyzeng/project/DiMTS/OUTPUT/fmri_seq256_pred128/ddpm_predict_fmri_seq256.npy")
b = np.load("/data/home/zyzeng/project/DiMTS/OUTPUT/fmri_seq256_pred128_ood/ddpm_predict_fmri_seq256_pred128_ood.npy")

print("shape:", a.shape, b.shape)
print("dtype:", a.dtype, b.dtype)

is_same = np.array_equal(a, b)
print("是否严格相同:", is_same)