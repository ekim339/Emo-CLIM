"""Script to extract wav files from m4a files."""


import os
import subprocess
import tqdm


# data paths:
SOURCE_DIR = "/Users/eugenekim/Emo-CLIM/dataset/AudioSet/eval_preprocess/eval_m4a"
TARGET_DIR = "/Users/eugenekim/Emo-CLIM/dataset/AudioSet/eval_preprocess/eval_wav_full"
# target sampling rate:
SAMPLE_RATE = 16000


if __name__ == "__main__":
    print("\n")

    # create target directory:
    os.makedirs(TARGET_DIR, exist_ok=True)

    # extract m4a file names:
    m4a_file_names = [name for name in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, name))]
    print("Number of m4a files: {}".format(len(m4a_file_names)))

    # extract wav files from m4a files:
    for m4a_file_name in tqdm.tqdm(m4a_file_names, total=len(m4a_file_names), desc="Extracting wav files..."):
        assert m4a_file_name.endswith(".m4a"), "File path does not end in '.m4a'"

        # create .wav file name:
        file_name = m4a_file_name.replace(".m4a", "")
        wav_file_name = file_name + ".wav"

        # call ffmpeg command:
        command = f"ffmpeg -i {os.path.join(SOURCE_DIR, m4a_file_name)} -ab 160k -ac 1 -ar {SAMPLE_RATE} -loglevel warning -vn {os.path.join(TARGET_DIR, wav_file_name)}"
        subprocess.call(command, shell=True)
    

    print("\n")

