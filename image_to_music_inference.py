"""Script to find top matching music for a given image."""

import os
import argparse
import torch
import torch.nn.functional as F
import clip
try:
    import laion_clap
except:
    pass
from PIL import Image
import tqdm

from climur.dataloaders.audioset import AudioSetMood
from climur.models.image_backbones import CLIPModel
from climur.models.audio_backbones import CLAPEmbeddings
from climur.trainers.image2music import Image2Music
from climur.utils.constants import (
    IMAGE2AUDIO_TAG_MAP,
    CLAP_INPUT_LENGTH,
    CLIP_EMBED_DIM,
    CLAP_EMBED_DIM,
)


def compute_single_image_embedding(model, image_path, image_preprocess_transform, device):
    """Compute embedding for a single image.
    
    Args:
        model: The trained Image2Music model
        image_path: Path to the image file
        image_preprocess_transform: CLIP preprocessing transform
        device: PyTorch device
    
    Returns:
        image_embed: Image embedding tensor
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = image_preprocess_transform(image)
    image = image.unsqueeze(0).to(device)
    
    # Compute embedding
    model.eval()
    with torch.no_grad():
        if model.multi_task:
            _, image_embed = model.compute_image_embeds(image)
        else:
            image_embed = model.compute_image_embeds(image)
    
    return image_embed.squeeze(0)


def get_audio_embeddings_and_paths(model, audio_dataset, device):
    """Extract all audio embeddings from dataset.
    
    Args:
        model: The trained Image2Music model
        audio_dataset: Audio dataset
        device: PyTorch device
    
    Returns:
        audio_embeds: List of audio embeddings
        audio_paths: List of audio file paths
        audio_tags: List of emotion tags
    """
    model.eval()
    audio_embeds = []
    audio_paths = []
    audio_tags = []
    
    for idx in tqdm.tqdm(range(len(audio_dataset)), desc="Extracting audio embeddings"):
        # Get audio chunks and metadata
        audio_chunks, tag = audio_dataset[idx]
        audio_chunks = audio_chunks.to(device)
        
        # Compute embedding
        with torch.no_grad():
            if model.multi_task:
                _, chunk_cross_embeds = model.compute_audio_embeds(audio_chunks)
                embed = chunk_cross_embeds.mean(dim=0)
            else:
                chunk_embeds = model.compute_audio_embeds(audio_chunks)
                embed = chunk_embeds.mean(dim=0)
        
        audio_embeds.append(embed)
        audio_tags.append(tag)
        
        # Construct the audio file path from metadata
        orig_subset = audio_dataset.metadata.loc[idx, "orig_subset"]
        file_name = audio_dataset.metadata.loc[idx, "file_name"]
        file_path = os.path.join(audio_dataset.root, audio_dataset.audio_dir_name, orig_subset, file_name)
        audio_paths.append(file_path)
    
    return audio_embeds, audio_paths, audio_tags


def find_top_matches(image_embed, audio_embeds, audio_paths, audio_tags, top_k=10):
    """Find top K matching audio files for the given image embedding.
    
    Args:
        image_embed: Image embedding tensor
        audio_embeds: List of audio embedding tensors
        audio_paths: List of audio file paths
        audio_tags: List of emotion tags
        top_k: Number of top matches to return
    
    Returns:
        top_matches: List of tuples (audio_path, similarity_score, emotion_tag)
    """
    # Stack audio embeddings into tensor
    audio_embeds_tensor = torch.stack(audio_embeds, dim=0)
    
    # Compute cosine similarity
    image_embed = image_embed.unsqueeze(0)
    cos_sim_scores = F.cosine_similarity(image_embed, audio_embeds_tensor, dim=-1)
    
    # Get top K matches
    top_k_values, top_k_indices = torch.topk(cos_sim_scores, k=min(top_k, len(audio_paths)))
    
    # Prepare results
    top_matches = []
    for i in range(len(top_k_indices)):
        idx = top_k_indices[i].item()
        similarity = top_k_values[i].item()
        top_matches.append((audio_paths[idx], similarity, audio_tags[idx]))
    
    return top_matches


def main():
    parser = argparse.ArgumentParser(description="Find top matching music for a given image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint_path", type=str, 
                       default="/Users/eugenekim/Emo-CLIM/train_logs/single_task/CLAP/frozen/all_losses/version_5/checkpoints/last.ckpt",
                       help="Path to model checkpoint")
    parser.add_argument("--audio_dataset_dir", type=str,
                       default="/Users/eugenekim/Emo-CLIM/dataset/AudioSet",
                       help="Path to audio dataset directory")
    parser.add_argument("--audio_metadata_file", type=str,
                       default="new_split_metadata_files/metadata_test.csv",
                       help="Audio dataset metadata file")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top matches to return")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID (-1 for CPU)")
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(f"cuda:{args.gpu}") if (args.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    print(f"\nUsing device: {device}\n")
    
    # Load CLIP model for image processing
    print("Loading CLIP model...")
    orig_clip_model, image_preprocess_transform = clip.load("ViT-B/32", device=device)
    image_backbone = CLIPModel(orig_clip_model)
    image_backbone.to(device)
    image_embed_dim = CLIP_EMBED_DIM
    
    # Load CLAP model for audio processing
    print("Loading CLAP model...")
    audio_clip_length = CLAP_INPUT_LENGTH
    audio_embed_dim = CLAP_EMBED_DIM
    full_audio_backbone = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    sample_audio_input = torch.rand((2, audio_clip_length))
    audio_backbone = CLAPEmbeddings(
        full_audio_backbone,
        sample_input=sample_audio_input,
        last_layer="layer7",
        pool_type="max",
    )
    audio_backbone.to(device)
    
    # Load trained Image2Music model
    print(f"Loading trained model from {args.checkpoint_path}...")
    full_model = Image2Music.load_from_checkpoint(
        args.checkpoint_path,
        image_backbone=image_backbone,
        audio_backbone=audio_backbone,
        output_embed_dim=128,
        image_embed_dim=image_embed_dim,
        audio_embed_dim=audio_embed_dim,
        multi_task=False,
        base_proj_hidden_dim=256,
        base_proj_dropout=0.2,
        base_proj_output_dim=128,
        task_proj_dropout=0.5,
        normalize_image_embeds=True,
        normalize_audio_embeds=True,
        freeze_image_backbone=True,
        freeze_audio_backbone=True,
        device=device
    )
    full_model.to(device)
    full_model.eval()
    
    # Compute embedding for query image
    print(f"\nProcessing query image: {args.image_path}")
    image_embed = compute_single_image_embedding(
        full_model, args.image_path, image_preprocess_transform, device
    )
    
    # Load audio dataset
    print(f"\nLoading audio dataset from {args.audio_dataset_dir}...")
    audio_dataset = AudioSetMood(
        root=args.audio_dataset_dir,
        metadata_file_name=args.audio_metadata_file,
        clip_length_samples=audio_clip_length,
        sample_rate=16000,
        augment_params=None,
        eval=True,
        overlap_ratio=0.75,
        audio_model="CLAP"
    )
    print(f"Found {len(audio_dataset)} audio files")
    
    # Extract audio embeddings
    print("\nExtracting audio embeddings...")
    audio_embeds, audio_paths, audio_tags = get_audio_embeddings_and_paths(
        full_model, audio_dataset, device
    )
    
    # Find top matches
    print(f"\nFinding top {args.top_k} matches...")
    top_matches = find_top_matches(image_embed, audio_embeds, audio_paths, audio_tags, args.top_k)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Top {args.top_k} matching music files for image: {args.image_path}")
    print(f"{'='*80}\n")
    
    for i, (audio_path, similarity, emotion_tag) in enumerate(top_matches, 1):
        print(f"{i}. Similarity: {similarity:.4f} | Emotion: {emotion_tag}")
        print(f"   File: {audio_path}\n")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

