import numpy as np

# Define a 4D array as a single line of code
tenny_4d_compact = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])

# Define a 4D array as a sequence of 3D arrays, with each set of brackets unfolded for clarity.

tenny_4d_expanded = np.array([
    [  # First 3D array in the sequence
        [  # First 2D table (slice) in the first 3D array
            [1, 2],  # First 1D row in the first 2D table
            [3, 4]   # Second 1D row in the first 2D table
        ],
        [  # Second 2D table (slice) in the first 3D array
            [5, 6],  # First 1D row in the second 2D table
            [7, 8]   # Second 1D row in the second 2D table
        ]
    ],
    [  # Second 3D array in the sequence
        [  # First 2D table (slice) in the second 3D array
            [9, 10],  # First 1D row in the first 2D table
            [11, 12]  # Second 1D row in the first 2D table
        ],
        [  # Second 2D table (slice) in the second 3D array
            [13, 14],  # First 1D row in the second 2D table
            [15, 16]   # Second 1D row in the second 2D table
        ]
    ]
    # More 3D arrays (sequence of 2D slices) could follow here
])

# Print the structured 4D array
# Print both 4D arrays to see the difference in structure visualization
print("Compact 4D Array:")
print(tenny_4d_compact)
print("\nExpanded 4D Array (for clarity):")
print(tenny_4d_expanded)