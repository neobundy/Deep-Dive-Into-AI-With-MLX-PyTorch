import numpy as np

# Simplified example for dimension reduction

# Original matrix with 100 rows and 10,000 columns
pretrained_llm = np.random.rand(100, 10000)  # High-dimensional data

# Apply SVD to the high-dimensional data
U, S, VT = np.linalg.svd(pretrained_llm, full_matrices=False)

# Reduce dimensions by selecting the top K singular values/vectors
# The `np.linalg.svd` function decomposes your original high-dimensional matrix `pretrained_llm` into three components:
# - `U`: A matrix whose columns are the left singular vectors
# - `S`: A diagonal matrix with singular values
# - `VT`: The transpose of a matrix whose rows are the right singular vectors
# To reduce the dimensions, you normally only keep the top `K` singular values (and corresponding singular vectors). The value `K` determines how many dimensions you want to keep.
# You can approximately reconstruct your matrix using only these top `K` components, which gives you a matrix that captures most of the important information from the original matrix but with the reduced dimensionality you desire.

K = 100  # Number of desired dimensions
U_reduced = U[:, :K]
S_reduced = np.diag(S[:K])
VT_reduced = VT[:K, :]

# Construct the reduced representation of the data
reduced_llm = np.dot(np.dot(U_reduced, S_reduced), VT_reduced)

# Shape of the reduced matrix
print("Shape of the original matrix:", pretrained_llm.shape)  # (100, 10000)

# However, the `reduced_llm` will still be the same shape as `pretrained_llm`. To truly reduce the dimensions of your data and work with a smaller matrix, you'd typically only use the `U_reduced` and `S_reduced`
# Now, `reduced_llm` actually is reduced, with 100 rows and `K` columns (in this case, 100). This smaller matrix is much easier to work with and can be used for further computation, analysis, or visualization.
print("Shape of the reduced data representation:", reduced_llm.shape)  # This will print (100, 10000)