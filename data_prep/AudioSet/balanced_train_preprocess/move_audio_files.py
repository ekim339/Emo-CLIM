"""Script to move already-cropped audio files to final location with correct naming."""

import os
import shutil

# Base paths:
ROOT = "/Users/eugenekim/Emo-CLIM/dataset/AudioSet"

# Subsets to process:
SUBSETS = {
    "balanced_train": {
        "source": os.path.join(ROOT, "balanced_train_preprocess/balanced_train_wav_full"),
        "target": os.path.join(ROOT, "audio_files/balanced_train")
    },
    "eval": {
        "source": os.path.join(ROOT, "eval_preprocess/eval_wav_full"),
        "target": os.path.join(ROOT, "audio_files/eval")
    }
}


def process_subset(source_dir, target_dir, subset_name):
    """Move and rename audio files for a given subset."""
    print(f"\n{'='*60}")
    print(f"Processing {subset_name} subset...")
    print(f"{'='*60}\n")
    
    # Create target directory:
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all WAV files:
    audio_files = [f for f in os.listdir(source_dir) 
                   if os.path.isfile(os.path.join(source_dir, f)) and f.endswith(".wav")]
    
    print(f"Found {len(audio_files)} audio files")
    
    copied_count = 0
    for filename in audio_files:
        # Parse filename: youtube_id_start-end.wav
        base_name = filename.replace(".wav", "")
        parts = base_name.rsplit("_", 1)
        
        if len(parts) == 2:
            youtube_id = parts[0]
            time_range = parts[1]
            
            # Extract start time from "start-end" format
            if "-" in time_range:
                start_time = time_range.split("-")[0]
            else:
                start_time = time_range
            
            # Create new filename: youtube_id_start.wav
            new_filename = f"{youtube_id}_{start_time}.wav"
            
            # Copy file to target location
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, new_filename)
            
            shutil.copy2(source_path, target_path)
            copied_count += 1
        else:
            print(f"Warning: Could not parse filename: {filename}")
    
    print(f"Successfully copied {copied_count} files to {target_dir}")


if __name__ == "__main__":
    print("\nMoving audio files to final locations...\n")
    
    # Process each subset:
    for subset_name, paths in SUBSETS.items():
        if os.path.exists(paths["source"]):
            process_subset(paths["source"], paths["target"], subset_name)
        else:
            print(f"\nWarning: Source directory not found for {subset_name}: {paths['source']}")
    
    print("\n" + "="*60)
    print("Done! All audio files have been moved to final locations.")
    print("="*60 + "\n")

