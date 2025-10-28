import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# data paths:
ORIG_SPLIT_METADATA_FILES = {
    "unbalanced_train": "/content/drive/MyDrive/LIKELION/EmoCLIM-dataset/AudioSet/metadata_unbalanced_train.csv",
    "balanced_train": "/content/drive/MyDrive/LIKELION/EmoCLIM-dataset/AudioSet/metadata_balanced_train.csv",
    "eval": "/content/drive/MyDrive/LIKELION/EmoCLIM-dataset/AudioSet/metadata_eval.csv"
}

# script options:
val_fract = 0.1
test_fract = 0.1
new_split_metadata_dir = "/content/drive/MyDrive/LIKELION/EmoCLIM-dataset/AudioSet/new_split_metadata_files"
new_split_metadata_files = {
    "train": "/content/drive/MyDrive/LIKELION/EmoCLIM-dataset/AudioSet/new_split_metadata_files/metadata_train.csv",
    "val": "/content/drive/MyDrive/LIKELION/EmoCLIM-dataset/AudioSet/new_split_metadata_files/metadata_val.csv",
    "test": "/content/drive/MyDrive/LIKELION/EmoCLIM-dataset/AudioSet/new_split_metadata_files/metadata_test.csv"
}

# load original split metadata files:
orig_split_metadata_dfs = {}
for subset, file_path in ORIG_SPLIT_METADATA_FILES.items():
    print("Loading {} set labels...".format(subset))
    orig_split_metadata_dfs[subset] = pd.read_csv(file_path)

# concatenate original split labels into a single dataframe:
all_metadata = pd.concat(orig_split_metadata_dfs.values(), axis="index")
all_metadata = all_metadata.reset_index(drop=True)
print()
print(all_metadata.info())

# print subset sizes:
train_set_size = int(np.around((1 - test_fract - val_fract) * all_metadata.shape[0]))
print("Training set size: {:.2f}% of dataset = {} samples.".format(100 * (1 - test_fract - val_fract), train_set_size))
val_set_size = int(np.around(val_fract * all_metadata.shape[0]))
print("Validation set size: {:.2f}% of dataset = {} samples.".format(100 * val_fract, val_set_size))
test_set_size = int(np.around(test_fract * all_metadata.shape[0]))
print("Test set size: {:.2f}% of dataset = {} samples.".format(100 * test_fract, test_set_size))

# get label counts:
label_counts = all_metadata["label"].value_counts()
for label in all_metadata["label"].value_counts().index:
    print("Number of {} clips: {}".format(label, label_counts[label]))

# split into stratified training/val and test sets:
metadata_train_val, metadata_test = train_test_split(all_metadata, test_size=test_set_size, stratify=all_metadata["label"], random_state=42)
assert metadata_test.shape[0] == test_set_size, "Test set metadata has incorrect size."
print(metadata_test.info())

# print test set class distribution:
print()
label_counts = metadata_test["label"].value_counts()
for label in metadata_test["label"].value_counts().index:
    print("Number of {} clips: {}".format(label, label_counts[label]))

# split into stratified training and validation sets:
metadata_train, metadata_val = train_test_split(metadata_train_val, test_size=val_set_size, stratify=metadata_train_val["label"], random_state=42)
assert metadata_val.shape[0] == val_set_size, "Validation set metadata has incorrect size."
print(metadata_val.info())

# print validation set class distribution:
print()
label_counts = metadata_val["label"].value_counts()
for label in metadata_val["label"].value_counts().index:
    print("Number of {} clips: {}".format(label, label_counts[label]))

# check that all subsets are disjoint:
metadata_subsets = [metadata_train, metadata_val, metadata_test]
subset_names = list(new_split_metadata_files.keys())
for subset_1, name_1 in zip(metadata_subsets, subset_names):
    for subset_2, name_2 in zip(metadata_subsets, subset_names):
        if name_1 != name_2:
            assert set(subset_1.index).isdisjoint(set(subset_2.index)), "{} and {} are not disjoint".format(name_1, name_2)

# reset indices:
metadata_train = metadata_train.reset_index(drop=True)
metadata_val = metadata_val.reset_index(drop=True)
metadata_test = metadata_test.reset_index(drop=True)

# sanity checks:
assert all_metadata.shape[0] == metadata_train.shape[0] + metadata_val.shape[0] + metadata_test.shape[0], "Subset set sizes don't add up."
# check that all subsets are disjoint:
metadata_subsets = [metadata_train, metadata_val, metadata_test]
subset_names = list(new_split_metadata_files.keys())
for subset_1, name_1 in zip(metadata_subsets, subset_names):
    for subset_2, name_2 in zip(metadata_subsets, subset_names):
        if name_1 != name_2:
            assert set(subset_1["file_name"].tolist()).isdisjoint(set(subset_2["file_name"].tolist())), "{} and {} are not disjoint".format(name_1, name_2)
# more sanity checks:
class_counts_all = all_metadata["label"].value_counts()
class_counts_train = metadata_train["label"].value_counts()
class_counts_val = metadata_val["label"].value_counts()
class_counts_test = metadata_test["label"].value_counts()
for class_label in all_metadata["label"].unique().tolist():
    assert class_counts_all[class_label] == class_counts_train[class_label] + class_counts_val[class_label] + class_counts_test[class_label], "Error with splitting dataset."

# save to file:
metadata_train.to_csv(new_split_metadata_files["train"], index=False)
metadata_val.to_csv(new_split_metadata_files["val"], index=False)
metadata_test.to_csv(new_split_metadata_files["test"], index=False)

