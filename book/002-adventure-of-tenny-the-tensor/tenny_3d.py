import numpy as np

# Initializing a 3D array in Python using NumPy.

# The 3D array can be visualized as a cube or a stack of tables (or 'shelves' of books in this analogy).
# Each pair of square brackets '[]' represents a shelf,
# and each shelf has its own set of tables or books organized in rows.

# Initialize a 3D array in a single line to represent the cube or stack of tables (compact form)
tenny_3d_compact = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Initialize the same 3D array with unfolded brackets for better readability (expanded form)
tenny_3d_expanded = np.array([
    [  # First 'shelf' or 2D layer
        [1, 2],  # First 1D row in the first 2D layer (a table on the first 'shelf')
        [3, 4]   # Second 1D row in the first 2D layer (another table on the first 'shelf')
    ],
    [  # Second 'shelf' or 2D layer
        [5, 6],  # First 1D row in the second 2D layer (a table on the second 'shelf')
        [7, 8]   # Second 1D row in the second 2D layer (another table on the second 'shelf')
    ]
    # Additional 'shelves' (2D layers) can be added here
])

# Print both 3D arrays to demonstrate the difference in visualization
print("Compact 3D Array:")
print(tenny_3d_compact)
print("\nExpanded 3D Array (for clarity):")
print(tenny_3d_expanded)