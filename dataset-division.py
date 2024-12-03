import os
import splitfolders

original_dataset_path = "./original_dataset"
classes = os.listdir(original_dataset_path)

output_folder = "./dataset"

splitfolders.ratio(
    original_dataset_path,
    output=output_folder,
    seed=42,
    ratio=(0.7, 0.2, 0.1),  
    group_prefix=None
)