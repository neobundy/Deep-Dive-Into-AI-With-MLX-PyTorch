import numpy as np
import torch

# ACT 1: THE ORDINARY WORLD - 0D
# Here we define Tenny as a zero-dimensional entity with a singular value
tenny_0d = np.array(5)
print("Act 1: Tenny in the ordinary 0D world:", tenny_0d)

# ACT 2: THE CALL TO ADVENTURE
# Tenny has no code change here, but realizes the potential for more dimensions

# ACT 3: CROSSING THE THRESHOLD - 1D
# Tenny is transformed into a 1-dimensional array
tenny_1d = np.array([5, 2, 3])
print("Act 3: Tenny crossing the threshold to 1D:", tenny_1d)

# ACT 4: THE PATH OF TRIALS - 2D
# Tenny now becomes a 2-dimensional array, a matrix
tenny_2d = np.array([[5, 2], [3, 4]])
print("Act 4: Tenny's path of trials in 2D:", tenny_2d)

# ACT 5: THE INNERMOST CAVE - 3D
# Tenny grows into a 3-dimensional array, a cube
tenny_3d = np.array([[[5], [2]], [[3], [4]]])
print("Act 5: Tenny in the innermost cave of 3D:", tenny_3d)

# ACT 6: THE ULTIMATE BOON - 4D
# Tenny now evolves into a 4-dimensional array, adding the concept of time, with the magic of torch
# Note that, strictly speaking, PyTorch is not necessary for this example. NumPy is more than enough. It's included merely for enjoyment.
tenny_4d = torch.tensor([[[[5], [2]], [[3], [4]]]])
print("Act 6: Tenny achieving the ultimate boon of 4D:", tenny_4d)

# ACT 7: RETURN WITH THE ELIXIR - To Infinity and Beyond
# Tenny has reached a new level of understanding and becomes a guide for others
# There's no additional transformation in this act, it's the culmination of Tenny's growth

# Let's print the dimensions of Tenny at the end of the journey to demonstrate Tenny's growth
print("Tenny's final form:")
print("Dimension:", tenny_4d.ndim)
print("Shape:", tenny_4d.shape)