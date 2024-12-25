import numpy as np

a_cat = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])

a_hat = np.array([[[28, 29, 30], [31, 32, 33], [34, 35, 36]], [[37, 38, 39], [40, 41, 42], [43, 44, 45]], [[46, 47, 48], [49, 50, 51], [52, 53, 54]]])

a_cat_with_hat = np.concatenate((a_cat, a_hat), axis=2)

print(f"A cat wearing a hat: {a_cat_with_hat}")

cuteness = np.array([[[55, 56, 57], [58, 59, 60], [61, 62, 63]], [[64, 65, 66], [67, 68, 69], [70, 71, 72]], [[73, 74, 75], [76, 77, 78], [79, 80, 81]]])

a_cute_cat = np.dot(a_cat, cuteness)

print(f"A cute cat wearing a hat: {a_cute_cat}")