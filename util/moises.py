import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
import time
from multiprocessing import Pool
from typing import List, Tuple, Dict, Optional, Union
import argparse

# HARDCODED PARAMETERS
SRC_DIR = "/home/phil/AudioSep/datasets/moisesdb/moisesdb_v0.1"
DEST_DIR = "/home/phil/AudioSep/datasets/moisesdb/moisesdb_compressed"
STEMS = ['bass', 'drums', 'vocals', 'guitar']
NUM_WORKERS = os.cpu_count()
MAX_FOLDERS = 240
TRIM = False
CREATE_VALID = False
VALID_DIR = "/home/phil/AudioSep/datasets/moisesdb/moisesdb_valid"
VALID_SIZE = 10
MIXTURE_NAME = "mixture.wav"


def combine_audio_files(files: List[str]) -> Tuple[np.ndarray, int]:
    """
    Combines multiple audio files into one by overlaying them.
    Parameters:
    - files (List[str]): List of file paths to be combined.
    Returns:
    - Tuple[np.ndarray, int]: A Tuple containing the combined audio data array and sample rate.
    """
    combined_data, sample_rate = sf.read(files[0])

    for file in files[1:]:
        data, sr = sf.read(file)
        if len(data) > len(combined_data):
            combined_data = np.pad(combined_data, ((0, len(data) - len(combined_data)), (0, 0)), 'constant')
        elif len(combined_data) > len(data):
            data = np.pad(data, ((0, len(combined_data) - len(data)), (0, 0)), 'constant')
        combined_data += data

    return combined_data, sample_rate


# Finds all .wav files located in folders that do not contain specified categories in their folder names
def files_to_categories(src_folder: str, categories: List[str]) -> Dict[str, List[str]]:
    """
    Finds all .wav files located in folders that do not contain specified categories
    in their folder names, within the given src_folder directory.
    Parameters:
    - src_folder (str): Path to the main directory containing subdirectories with files.
    - categories (List[str]): Keywords that should not be part of the folder's name.
    Returns:
    - Dict[str, List[str]]: A Dict with keys as categories, values as lists of paths to .wav files found.
    """
    files = {category: [] for category in categories + ['other']}

    for folder in os.listdir(src_folder):
        folder_path = os.path.join(src_folder, folder)
        if os.path.isdir(folder_path):
            if folder.lower() in categories:
                stem = folder.lower()
            else:
                stem = 'other'
            for f in os.listdir(folder_path):
                if f.endswith('.wav'):
                    files[stem].append(os.path.join(folder_path, f))
    return files


# Processes a folder containing audio tracks, copying and combining necessary files into the target structure
def process_folder(src_folder: str, dest_folder: str, stems: List[str], trim: bool = False) -> None:
    """
    Processes a folder containing audio tracks, copying and combining necessary files into the target structure.
    Parameters:
    - src_folder (str): Path to the source folder of MoisesDB.
    - dest_folder (str): Path to the target folder for MUSDB18.
    - stems (List[str]): List of stem categories to process.
    - trim (bool): If True, trim all stems to the length of the shortest one.
    """

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    categories = stems

    if trim:
        # First pass: load all stems and find the minimum length
        stem_data = {}
        sample_rate = None
        min_length = float('inf')
        all_files = files_to_categories(src_folder, categories)

        # Using tqdm to display progress for categories
        for category in tqdm(categories, desc=f"Processing categories in {os.path.basename(src_folder)}"):
            files = all_files[category]
            if files:
                if len(files) > 1:
                    combined_data, sr = combine_audio_files(files)
                else:
                    combined_data, sr = sf.read(files[0])

                stem_data[category] = combined_data
                sample_rate = sr
                min_length = min(min_length, len(combined_data))

        # Process 'other' files
        other_files = all_files['other']
        if other_files:
            other_combined_data, sr = combine_audio_files(other_files)
            stem_data['other'] = other_combined_data
            sample_rate = sr
            min_length = min(min_length, len(other_combined_data))

        # If no stems were found, set a default sample rate
        if sample_rate is None:
            sample_rate = 44100
            min_length = 0

        # Second pass: trim all stems to min_length and save
        for category in categories:
            if category in stem_data:
                trimmed_data = stem_data[category][:min_length]
                sf.write(os.path.join(dest_folder, f"{category}.wav"), trimmed_data, sample_rate)
            else:
                # Create silence with min_length
                silence = np.zeros((min_length, 2), dtype=np.float32)
                sf.write(os.path.join(dest_folder, f"{category}.wav"), silence, sample_rate)

        # Save 'other' stem
        if 'other' in stem_data:
            trimmed_other = stem_data['other'][:min_length]
            sf.write(os.path.join(dest_folder, "other.wav"), trimmed_other, sample_rate)
        else:
            silence = np.zeros((min_length, 2), dtype=np.float32)
            sf.write(os.path.join(dest_folder, "other.wav"), silence, sample_rate)

        # Create mixture.wav from all files, then trim to min_length
        all_files_list = [file for sublist in all_files.values() for file in sublist]
        if all_files_list:
            mixture_data, sample_rate = combine_audio_files(all_files_list)
            mixture_data = mixture_data[:min_length]
            sf.write(os.path.join(dest_folder, "mixture.wav"), mixture_data, sample_rate)
        else:
            # If no files at all, create silent mixture
            silence = np.zeros((min_length, 2), dtype=np.float32)
            sf.write(os.path.join(dest_folder, "mixture.wav"), silence, sample_rate)
    else:
        # Original behavior: If the required stem does not exist in the source folder (src_folder),
        # we add silence instead of the file with the same duration as the standard file.
        problem_categories = []
        duration = 0
        all_files = files_to_categories(src_folder, categories)

        # Using tqdm to display progress for categories
        for category in tqdm(categories, desc=f"Processing categories in {os.path.basename(src_folder)}"):
            files = all_files[category]
            if files:
                if len(files) > 1:
                    combined_data, sample_rate = combine_audio_files(files)
                else:
                    combined_data, sample_rate = sf.read(files[0])

                sf.write(os.path.join(dest_folder, f"{category}.wav"), combined_data, sample_rate)
                duration = max(duration, len(combined_data) / sample_rate)
            else:
                problem_categories.append(category)

        other_files = all_files['other']
        if other_files:
            other_combined_data, sample_rate = combine_audio_files(other_files)
            sf.write(os.path.join(dest_folder, "other.wav"), other_combined_data, sample_rate)
        else:
            problem_categories.append('other')

        for category in problem_categories:
            silence = np.zeros((int(duration * sample_rate), 2), dtype=np.float32)
            sf.write(os.path.join(dest_folder, f"{category}.wav"), silence, sample_rate)
        # mixture.wav
        all_files_list = [file for sublist in all_files.values() for file in sublist]
        mixture_data, sample_rate = combine_audio_files(all_files_list)
        sf.write(os.path.join(dest_folder, "mixture.wav"), mixture_data, sample_rate)


# Wrapper function for 'process_folder' that unpacks the arguments
def process_folder_wrapper(args: Tuple[str, str, List[str], bool]) -> None:
    """
    A wrapper function for 'process_folder' that unpacks the arguments.
    Parameters:
    - args (Tuple[str, str, List[str], bool]): A Tuple containing the source folder, destination folder paths, stems, and trim flag.
    """
    src_folder, dest_folder, stems, trim = args
    return process_folder(src_folder, dest_folder, stems, trim)


# Converts MoisesDB dataset to MUSDB18 format for a specified number of folders
def convert_dataset(src_root: str, dest_root: str, stems: List[str], max_folders: int = 240, num_workers: int = 4,
                    trim: bool = False) -> None:
    """
    Converts MoisesDB dataset to MUSDB18 format for a specified number of folders.
    Parameters:
    - src_root (str): Root directory of the MoisesDB dataset.
    - dest_root (str): Root directory where the new dataset will be saved.
    - max_folders (int): Maximum number of folders to process.
    - num_workers (int): Number of parallel workers for processing.
    - trim (bool): If True, trim all stems to the length of the shortest one.
    """
    folders_to_process = []
    for folder in os.listdir(src_root):
        if len(folders_to_process) >= max_folders:
            break

        src_folder = os.path.join(src_root, folder)
        dest_folder = os.path.join(dest_root, folder)

        if os.path.isdir(src_folder):
            folders_to_process.append((src_folder, dest_folder, stems, trim))
        else:
            print(f"Skip {src_folder} — not dir")

    with Pool(num_workers) as pool:
        pool.map(process_folder_wrapper, folders_to_process)


# Count number of subfolders in a folder
def count_folders_in_folder(args_to_func) -> Dict[str, int]:
    """
    Counts the number of subfolders inside a given folder.

    Parameters:
    - folder_path (str): Path to the folder where the count is needed.

    Returns:
    - Dict[str, int]: A dictionary with folder paths as keys and subfolder counts as values.
    """
    folder_count = 0
    folder_path, stems = args_to_func
    if os.path.isdir(folder_path):
        # Count subfolders in stems
        folder_count = len([f for f in os.listdir(folder_path)
                            if os.path.isdir(os.path.join(folder_path, f)) and f in stems])
        # For other.wav
        if any(os.path.isdir(os.path.join(folder_path, f)) and f not in stems for f in os.listdir(folder_path)):
            folder_count += 1

    return {folder_path: folder_count}


# Parallel count of subfolders in each folder inside src_folder
def count_folders_parallel(src_folder: str, stems, num_workers: int = 4) -> Dict[str, int]:
    """
    Parallelly counts the number of subfolders in each folder inside src_folder.

    Parameters:
    - src_folder (str): Root folder containing subfolders to count.

    Returns:
    - Dict[str, int]: Dictionary with folder paths as keys and subfolder counts as values.
    """
    # Get list of all folders inside src_folder
    folders_to_process = [os.path.join(src_folder, folder) for folder in os.listdir(src_folder) if
                          os.path.isdir(os.path.join(src_folder, folder))]

    args_to_func = [(folder, stems) for folder in folders_to_process]

    # Parallelly process each folder using pool.map
    with Pool(num_workers) as pool:
        results = pool.map(count_folders_in_folder, args_to_func)

    # Merge results from different processes
    merged_counts = {}
    for result in results:
        merged_counts.update(result)

    return merged_counts


def main() -> None:
    start = time.time()

    print(f'Configuration:')
    print(f'  src_dir: {SRC_DIR}')
    print(f'  dest_dir: {DEST_DIR}')
    print(f'  stems: {STEMS}')
    print(f'  num_workers: {NUM_WORKERS}')
    print(f'  max_folders: {MAX_FOLDERS}')
    print(f'  trim: {TRIM}')
    print(f'  create_valid: {CREATE_VALID}')
    print(f'  categories: {STEMS + ["other"]}')
    print()

    convert_dataset(SRC_DIR, DEST_DIR, STEMS, max_folders=MAX_FOLDERS, num_workers=NUM_WORKERS, trim=TRIM)

    print(f'All {MAX_FOLDERS} files have been processed, time: {time.time() - start:.2f} sec')

    if CREATE_VALID:
        # Count folders in the MoisesDB dataset
        result = count_folders_parallel(SRC_DIR, STEMS)
        result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        valid_size = VALID_SIZE
        list_folders = list(result.keys())
        print(f'Top {valid_size} folders:')
        for track in list(result.items())[:valid_size]:
            print(track)

        valid_folder = VALID_DIR
        train_tracks_folder = DEST_DIR

        # Create valid folder if not exists
        if not os.path.exists(valid_folder):
            os.makedirs(valid_folder)

        num_val = 0

        # Copy folders from train_tracks to valid folder
        for folder in list_folders:
            folder_name = os.path.basename(folder)  # Get folder name

            # Form the path to the folder in train_tracks
            source_folder = os.path.join(train_tracks_folder, folder_name)

            # If the folder exists in train_tracks, copy it to valid
            if os.path.exists(source_folder):
                destination = os.path.join(valid_folder, folder_name)
                shutil.copytree(source_folder, destination)
                shutil.rmtree(source_folder)
                num_val += 1
                print(f'Folder: {folder}, num_stems: {result[folder]}')
                if num_val >= valid_size:
                    break
            else:
                print(f"Folder {source_folder} not found.")

    print('The end!')


if __name__ == '__main__':
    main()
