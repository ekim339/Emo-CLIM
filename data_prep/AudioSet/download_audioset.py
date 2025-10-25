from audioset_download import Downloader
import os

# choose where files go
ROOT = "/Users/eugenekim/Emo-CLIM/dataset/AudioSet" 

# pick labels (names as in AudioSet ontology)
labels = ["Happy music", "Funny music", "Sad music", "Tender music", "Exciting music", "Angry music", "Scary music"]   # <- example; replace with what you need

d_balanced_train = Downloader(
    root_path=os.path.join(ROOT, "balanced_train_preprocess/balanced_train_m4a"),
    labels=labels,                # None => all labels
    n_jobs=4,                     # parallel downloads
    download_type="balanced_train",  # 'balanced_train' | 'unbalanced_train' | 'eval'
    copy_and_replicate=False      # if a clip has multiple labels, store once
)

# choose an audio format supported by yt-dlp’s --audio-format
d_balanced_train.download(format="m4a")          # e.g., "vorbis" (default), "mp3", "m4a", "flac", "opus", "webm"

d_eval = Downloader(
    root_path=os.path.join(ROOT, "eval_preprocess/eval_m4a"),
    labels=labels,                # None => all labels
    n_jobs=4,                     # parallel downloads
    download_type="eval",  # 'balanced_train' | 'unbalanced_train' | 'eval'
    copy_and_replicate=False      # if a clip has multiple labels, store once
)

d_eval.download(format="m4a")          # e.g., "vorbis" (default), "mp3", "m4a", "flac", "opus", "webm"

# d_unbalanced_train = Downloader(
#     root_path=os.path.join(ROOT, "unbalanced_train_preprocess/unbalanced_train_m4a"),
#     labels=labels,                # None => all labels
#     n_jobs=4,                     # parallel downloads
#     download_type="unbalanced_train",  # 'balanced_train' | 'unbalanced_train' | 'eval'
#     copy_and_replicate=False      # if a clip has multiple labels, store once
# )

# os.makedirs(ROOT, exist_ok=True)
# os.makedirs(os.path.join(ROOT, "unbalanced_train_preprocess/unbalanced_train_m4a"), exist_ok=True)
# # choose an audio format supported by yt-dlp’s --audio-format
# d_unbalanced_train.download(format="m4a") 

# Flatten directory structure - move all files from subdirectories to parent
import shutil

def flatten_directory(parent_dir):
    """Move all files from subdirectories to parent directory."""
    for subdir in os.listdir(parent_dir):
        subdir_path = os.path.join(parent_dir, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                if os.path.isfile(file_path):
                    dest_path = os.path.join(parent_dir, filename)
                    shutil.move(file_path, dest_path)
                    print(f"Moved: {filename}")
            # Remove empty subdirectory
            os.rmdir(subdir_path)
            print(f"Removed empty directory: {subdir}")

print("\nFlattening balanced_train directory...")
flatten_directory(os.path.join(ROOT, "balanced_train_preprocess/balanced_train_m4a"))

print("\nFlattening eval directory...")
flatten_directory(os.path.join(ROOT, "eval_preprocess/eval_m4a"))

# print("\nFlattening balanced_train directory...")
# flatten_directory(os.path.join(ROOT, "unbalanced_train_preprocess/balanced_train_m4a"))

print("\nDone! All files are now in the parent directories.")
