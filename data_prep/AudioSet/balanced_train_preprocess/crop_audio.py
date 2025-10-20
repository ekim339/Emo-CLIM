"""Script to extract desired segments from audio files."""


import os
import torchaudio
import tqdm


# data paths:
SOURCE_DIR = "/Users/eugenekim/Emo-CLIM/dataset/AudioSet/balanced_train_preprocess/balanced_train_wav_full"
TARGET_DIR = "/Users/eugenekim/Emo-CLIM/dataset/AudioSet/audio_files/balanced_train"

# constants:
SAMPLE_RATE = 16000
CLIP_LENGTH_SEC = 10.0
CLIP_LENGTH_SAMPLES = int(CLIP_LENGTH_SEC * SAMPLE_RATE)


if __name__ == "__main__":
    print("\n")
    
    # create target directory:
    os.makedirs(TARGET_DIR, exist_ok=True)

    # extract audio file names:
    audio_file_names = [name for name in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, name))]
    print("Number of audio files: {}".format(len(audio_file_names)))

    # extract desired segments from full audio files:
    n_unexpect_files = 0
    min_clip_length = CLIP_LENGTH_SAMPLES
    print()
    for file_name in tqdm.tqdm(audio_file_names, total=len(audio_file_names), desc="Extracting desired segments..."):
        # sanity check:
        assert file_name.endswith(".wav"), "File name does not end in '.wav'"

        # extract youtube ID from file name:
        youtube_id_start_time = file_name.replace(".wav", "")
        assert len(youtube_id_start_time) == len(file_name) - len(".wav"), "Error with removing '.wav'."
        start_time_str = youtube_id_start_time.split("_")[-1]
        youtube_id = youtube_id_start_time.replace("_" + start_time_str, "")
        assert len(youtube_id) == len(youtube_id_start_time) - len(start_time_str) - 1, "Error with removing start_time."
        # convert start time string to float (handle "start-end" format):
        if "-" in start_time_str:
            start_time_sec = float(start_time_str.split("-")[0])  # Extract start time from "start-end"
        else:
            start_time_sec = float(start_time_str)

        # load audio:
        audio_full, sample_rate = torchaudio.load(os.path.join(SOURCE_DIR, file_name))
        audio_full = audio_full.squeeze(dim=0)
        assert sample_rate == SAMPLE_RATE, "Unexpected sampling rate."

        # extract desired segment:
        start_time_samples = int(start_time_sec * SAMPLE_RATE)
        audio_segment = audio_full[start_time_samples : start_time_samples + CLIP_LENGTH_SAMPLES]
        audio_segment = audio_segment.unsqueeze(dim=0)
        
        # Skip if segment is too short or empty:
        if audio_segment.size(dim=-1) < CLIP_LENGTH_SAMPLES:
            n_unexpect_files += 1
            if audio_segment.size(dim=-1) < min_clip_length:
                min_clip_length = audio_segment.size(dim=-1)
            # Skip saving this file if it's too short
            continue
        
        # save to file:
        save_file_name = f"{youtube_id}_{start_time_sec}.wav"
        torchaudio.save(
            os.path.join(TARGET_DIR, save_file_name),  # First positional argument (no 'filepath=')
            audio_segment,
            SAMPLE_RATE,
            channels_first=True
        )
    
    print("\nNumber of files with unexpected lengths: {}".format(n_unexpect_files))
    print("Minimim clip length: {}s".format(min_clip_length / SAMPLE_RATE))


    print("\n")

