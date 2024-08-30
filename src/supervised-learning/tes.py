import numpy as np

def group_by_element(arr):
  unique_elements, indices = np.unique(arr, return_index=True)
  grouped_elements = {}
  for elem, idx in zip(unique_elements, indices):
    grouped_elements[elem] = list(np.where(arr == elem)[0])
  return grouped_elements

# Example usage:
arr = np.array([1, 2, 1, 3, 2, 1])
result = group_by_element(arr)
print(result)