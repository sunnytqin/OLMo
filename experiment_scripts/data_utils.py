"""Utility functions for counting tokens and managing data shards.

=============================================================================
PRETRAINING DATA MIX (20B Tokens - Balanced Mix for 1B Model)
=============================================================================

MATH (non-GSM) - Keep All (~3.67 BT = 18.4%):
- mathcoder2-synthmath (filtered-math subdirectory) → 3.09 BT (keep all)
- mathcoder2-synthmath (ajibawa-2023 subdirectory) → 782.58 MT (keep all)
- personahub_math_v5_regen_149960 → 191.58 MT (keep all)
- owm-filtered-math (metamath subdirectory) → 84.22 MT (keep all)
- tulu-3-sft-personas-math-grade → 21.80 MT (keep all)
- tulu_v3.9_personahub_math_interm_algebra_20k → 19.74 MT (keep all)
- basic_math_mj → 9.03 MT (keep all)
- basic_math_mj (multiadd subdirectory) → 2.21 MT (keep all)

GENERAL/WEB - Target 11 BT (~55%):
- dclm → 24.31 BT → 8.5 BT (35% retention)
- olmo-mix (wiki subdirectory) → 3.66 BT → 2.5 BT (68.3% retention)

SCIENTIFIC/ACADEMIC - Target 3.0 BT (~15%):
- pes2o → 3.01 BT → 2.1 BT (69.8% retention)
- stackexchange → 1.26 BT → 0.9 BT (71.4% retention)

INSTRUCTION/REASONING - Target 2.33 BT (~11.6%):
- tulu_flan → 8.54 BT → 2.33 BT (27.3% retention)

EXCLUDED FROM PRETRAINING (GSM-related):
- gsm8k (all variants)
- gsm8k-synth
- gsm_MIND
- tinyGSM (all variants)

Total Pretraining: ~20 BT
Breakdown: 18.4% Math, 55% General/Web, 15% Scientific, 11.6% Instruction

=============================================================================
SFT DATA MIX (20.88M Tokens - GSM8K-focused)
=============================================================================

GSM8K DATASETS (excluded from pretraining, used for SFT):
- gsm_MIND/clean_stop → 17.06 MT (81.7%) - 92 shards
- gsm8k/v0_socratic_train → 1.51 MT (7.3%) - 1 shard
- gsm8k/v0_main_train → 1.23 MT (5.9%) - 1 shard
- gsm8k-synth/resample_v1_6x → 1.08 MT (5.2%) - 1 shard

Total SFT Dataset: 20.88 MT (95 shards)

EXCLUDED FROM SFT (too large):
- tinyGSM/mind → 3.06 BT
- tinyGSM/mind-2students → 3.41 BT
"""

import numpy as np
from pathlib import Path
from typing import List, Union
import glob
import yaml


def count_tokens(shard_paths: Union[str, List[str]], dtype=np.uint32) -> int:
    """
    Count total number of tokens across one or more shard files.

    Args:
        shard_paths: A single shard path or list of shard paths
        dtype: The numpy dtype of the memmap (default: np.uint32)

    Returns:
        Total number of tokens across all shards

    Example:
        >>> count_tokens('/path/to/shard.npy')
        26186464
        >>> count_tokens(['/path/to/shard1.npy', '/path/to/shard2.npy'])
        52372928
    """
    if isinstance(shard_paths, str):
        shard_paths = [shard_paths]

    total_tokens = 0
    for shard_path in shard_paths:
        arr = np.memmap(shard_path, dtype=dtype, mode='r')
        total_tokens += len(arr)

    return total_tokens


def get_local_shards(source_name: str,
                     base_dir: str = "/n/netscratch/dam_lab/Lab/sqin/olmo/stage2/preprocessed",
                     tokenizer: str = "dolma2-tokenizer",
                     recursive: bool = True) -> List[str]:
    """
    Get all local shard files for a given data source.

    Args:
        source_name: Name of the data source (e.g., 'personahub_math_v5_regen_149960')
        base_dir: Base directory where preprocessed data is stored
        tokenizer: Tokenizer subdirectory name (default: 'dolma2-tokenizer')
        recursive: If True, search recursively for .npy files (default: True)

    Returns:
        Sorted list of absolute paths to all shard files for this source

    Example:
        >>> shards = get_local_shards('personahub_math_v5_regen_149960')
        >>> len(shards)
        15
        >>> shards[0]
        '/n/netscratch/dam_lab/Lab/sqin/olmo/stage2/preprocessed/personahub_math_v5_regen_149960/dolma2-tokenizer/part-00-00000.npy'
    """
    if tokenizer:
        source_dir = Path(base_dir) / source_name / tokenizer
    else:
        source_dir = Path(base_dir) / source_name

    if not source_dir.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    # Find all .npy files in the directory (recursively if specified)
    if recursive:
        shard_files = sorted(glob.glob(str(source_dir / "**" / "*.npy"), recursive=True))
    else:
        shard_files = sorted(glob.glob(str(source_dir / "*.npy")))

    if not shard_files:
        raise ValueError(f"No .npy files found in {source_dir}")

    return shard_files


def get_source_token_count(source_name: str,
                          base_dir: str = "/n/netscratch/dam_lab/Lab/sqin/olmo/stage2/preprocessed",
                          tokenizer: str = "dolma2-tokenizer",
                          dtype=np.uint32) -> int:
    """
    Get total token count for an entire data source.

    Args:
        source_name: Name of the data source
        base_dir: Base directory where preprocessed data is stored
        tokenizer: Tokenizer subdirectory name (default: 'dolma2-tokenizer')
        dtype: The numpy dtype of the memmap (default: np.uint32)

    Returns:
        Total number of tokens in the data source

    Example:
        >>> get_source_token_count('personahub_math_v5_regen_149960')
        191583960
    """
    shards = get_local_shards(source_name, base_dir, tokenizer)
    return count_tokens(shards, dtype)


def select_shards_for_target(source_name: str,
                            target_tokens: int,
                            base_dir: str = "/n/netscratch/dam_lab/Lab/sqin/olmo/stage2/preprocessed",
                            tokenizer: str = "dolma2-tokenizer",
                            dtype=np.uint32) -> tuple[List[str], int]:
    """
    Select shards from a source until reaching the target token count.

    Args:
        source_name: Name of the data source
        target_tokens: Target number of tokens to select
        base_dir: Base directory where preprocessed data is stored
        tokenizer: Tokenizer subdirectory name (default: 'dolma2-tokenizer')
        dtype: The numpy dtype of the memmap (default: np.uint32)

    Returns:
        Tuple of (selected_shard_paths, actual_token_count)
        - selected_shard_paths: List of shard paths that were selected
        - actual_token_count: Actual number of tokens in selected shards (may slightly exceed target)

    Example:
        >>> shards, tokens = select_shards_for_target('dclm', 8_500_000_000)
        >>> print(f"Selected {len(shards)} shards with {tokens:,} tokens")
    """
    all_shards = get_local_shards(source_name, base_dir, tokenizer)

    selected_shards = []
    total_tokens = 0

    for shard_path in all_shards:
        shard_tokens = count_tokens(shard_path, dtype)
        selected_shards.append(shard_path)
        total_tokens += shard_tokens

        if total_tokens >= target_tokens:
            break

    return selected_shards, total_tokens


def build_data_mix(base_dir: str = "/n/netscratch/dam_lab/Lab/sqin/olmo/stage2/preprocessed",
                   tokenizer: str = "dolma2-tokenizer",
                   dtype=np.uint32) -> List[str]:
    """
    Build the complete data mix according to the target specification.

    Returns:
        List of all selected shard paths

    Prints a detailed summary of:
        - Token counts per source
        - Token counts per category
        - Total token count
    """

    # Define targets based on the specification at the top of this file
    # Format: (source_name, target_tokens_in_billions or millions, category)

    targets = [
        # MATH (non-GSM) - Keep All
        ("mathcoder2-synthmath/mathcoder2-synthmath/filtered-math", 3.09e9, "MATH"),
        ("mathcoder2-synthmath/ajibawa-2023", 782.58e6, "MATH"),
        ("personahub_math_v5_regen_149960", 191.58e6, "MATH"),
        ("owm-filtered-math/metamath", 84.22e6, "MATH", ""),  # No dolma2-tokenizer subdirectory
        ("tulu-3-sft-personas-math-grade", 21.80e6, "MATH"),
        ("tulu_v3.9_personahub_math_interm_algebra_20k", 19.74e6, "MATH"),
        ("basic_math_mj", 9.03e6, "MATH"),
        ("basic_math_mj/multiadd", 2.21e6, "MATH"),

        # GENERAL/WEB
        ("dclm/v0_rep32_ft7percentile_fw2/documents/allenai", 8.5e9, "GENERAL/WEB"),
        ("olmo-mix/danyh-compiled-v1_7/documents/wiki/allenai", 2.5e9, "GENERAL/WEB"),

        # SCIENTIFIC/ACADEMIC
        ("pes2o/allenai", 2.1e9, "SCIENTIFIC/ACADEMIC"),
        ("stackexchange/v1_dedupe/allenai", 0.9e9, "SCIENTIFIC/ACADEMIC"),

        # INSTRUCTION/REASONING
        ("tulu_flan/v1-FULLDECON-HARD-TRAIN-60M-shots_all-upweight_1-dialog_false-sep_rulebased/allenai", 2.33e9, "INSTRUCTION/REASONING"),
    ]

    all_selected_shards = []
    results_by_source = []
    category_totals = {}

    print("Building Data Mix...")
    print("=" * 80)
    print()

    for target in targets:
        # Handle both 3-tuple and 4-tuple formats
        if len(target) == 4:
            source_name, target_tokens, category, custom_tokenizer = target
        else:
            source_name, target_tokens, category = target
            custom_tokenizer = tokenizer

        try:
            selected_shards, actual_tokens = select_shards_for_target(
                source_name,
                int(target_tokens),
                base_dir,
                custom_tokenizer,
                dtype
            )

            all_selected_shards.extend(selected_shards)
            results_by_source.append((source_name, category, target_tokens, actual_tokens, len(selected_shards)))

            # Accumulate category totals
            if category not in category_totals:
                category_totals[category] = 0
            category_totals[category] += actual_tokens

            print(f"✓ {source_name}")
            print(f"  Target: {target_tokens/1e9:.2f}B | Actual: {actual_tokens/1e9:.2f}B | Shards: {len(selected_shards)}")
            print()

        except Exception as e:
            print(f"✗ {source_name}: ERROR - {e}")
            print()

    # Print summary by category
    print("=" * 80)
    print("CATEGORY SUMMARY:")
    print("=" * 80)

    grand_total = 0
    for category in ["MATH", "GENERAL/WEB", "SCIENTIFIC/ACADEMIC", "INSTRUCTION/REASONING"]:
        if category in category_totals:
            tokens = category_totals[category]
            grand_total += tokens
            print(f"{category:30s}: {tokens/1e9:6.2f}B tokens")

    print("-" * 80)
    print(f"{'TOTAL':30s}: {grand_total/1e9:6.2f}B tokens")
    print("=" * 80)
    print()

    # Print detailed breakdown
    print("DETAILED SOURCE BREAKDOWN:")
    print("=" * 80)
    for source_name, category, target, actual, num_shards in results_by_source:
        pct = (actual / grand_total * 100) if grand_total > 0 else 0
        print(f"{source_name:70s} | {actual/1e9:6.2f}B ({pct:4.1f}%) | {num_shards:4d} shards")

    print("=" * 80)
    print(f"Total shards selected: {len(all_selected_shards)}")
    print()

    return all_selected_shards


def build_gsm8k_sft_mix(base_dir: str = "/n/netscratch/dam_lab/Lab/sqin/olmo/stage2/preprocessed",
                        dtype=np.uint32) -> List[str]:
    """
    Build GSM8K SFT data mix (excludes large tinyGSM datasets).

    Includes:
    - gsm8k/v0_main_train (1.23M tokens)
    - gsm8k/v0_socratic_train (1.51M tokens)
    - gsm8k-synth/resample_v1_6x (1.08M tokens)
    - gsm_MIND/clean_stop (17.06M tokens)

    Total: ~20.88M tokens across 95 shards

    Returns:
        List of all GSM8K shard paths
    """

    # Define GSM8K sources (small datasets only, excluding tinyGSM)
    gsm_sources = [
        ("gsm8k/v0_main_train/allenai", "dolma2-tokenizer"),
        ("gsm8k/v0_socratic_train/allenai", "dolma2-tokenizer"),
        ("gsm8k-synth/resample_v1_6x", "dolma2-tokenizer"),
        ("gsm_MIND/clean_stop", "dolma2-tokenizer"),
    ]

    all_shards = []
    results = []

    print("Building GSM8K SFT Data Mix...")
    print("=" * 80)
    print()

    for source_name, tokenizer in gsm_sources:
        try:
            shards = get_local_shards(source_name, base_dir, tokenizer)
            tokens = count_tokens(shards, dtype)

            all_shards.extend(shards)
            results.append((source_name, tokens, len(shards)))

            print(f"✓ {source_name}")
            print(f"  Tokens: {tokens/1e6:.2f}M | Shards: {len(shards)}")
            print()

        except Exception as e:
            print(f"✗ {source_name}: ERROR - {e}")
            print()

    # Print summary
    total_tokens = sum(r[1] for r in results)
    total_shards = sum(r[2] for r in results)

    print("=" * 80)
    print("GSM8K SFT DATA SUMMARY:")
    print("=" * 80)
    for source_name, tokens, num_shards in results:
        pct = (tokens / total_tokens * 100) if total_tokens > 0 else 0
        print(f"{source_name:60s} | {tokens/1e6:6.2f}M ({pct:5.1f}%) | {num_shards:4d} shards")

    print("-" * 80)
    print(f"{'TOTAL':60s} | {total_tokens/1e6:6.2f}M tokens | {total_shards:4d} shards")
    print("=" * 80)
    print()

    return all_shards


def local_to_remote_path(local_path: str,
                        local_base: str = "/n/netscratch/dam_lab/Lab/sqin/olmo/stage2/preprocessed",
                        remote_base: str = "http://olmo-data.org/preprocessed") -> str:
    """
    Convert a local file path to its remote URL.

    Args:
        local_path: Local file path
        local_base: Base directory for local files
        remote_base: Base URL for remote files

    Returns:
        Remote URL

    Example:
        >>> local_to_remote_path('/n/netscratch/.../preprocessed/dclm/part-0.npy')
        'http://olmo-data.org/preprocessed/dclm/part-0.npy'
    """
    # Remove the local base and replace with remote base
    relative_path = local_path.replace(local_base, "").lstrip("/")
    return f"{remote_base}/{relative_path}"


def write_paths_to_yaml(shard_paths: List[str],
                       yaml_path: str,
                       use_remote: bool = False,
                       local_base: str = "/n/netscratch/dam_lab/Lab/sqin/olmo/stage2/preprocessed",
                       remote_base: str = "http://olmo-data.org/preprocessed") -> None:
    """
    Write shard paths to the YAML config file under data.paths.

    Args:
        shard_paths: List of local shard paths
        yaml_path: Path to the YAML config file
        use_remote: If True, convert to remote URLs; if False, use local paths (default: False)
        local_base: Base directory for local files
        remote_base: Base URL for remote files
    """
    # Convert to remote URLs if requested, otherwise use local paths
    if use_remote:
        paths = [local_to_remote_path(path, local_base, remote_base) for path in shard_paths]
    else:
        paths = shard_paths

    # Load the existing YAML config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update the data.paths field
    if 'data' not in config:
        config['data'] = {}

    config['data']['paths'] = paths

    # Write back to the YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=1000)

    path_type = "remote URLs" if use_remote else "local paths"
    print(f"✓ Updated {yaml_path}")
    print(f"  Added {len(paths)} shard {path_type} to data.paths")


def build_and_write_config(yaml_path: str = "/n/home05/sqin/OLMo/experiment_scripts/OLMo2-1B-stage1.yaml",
                          base_dir: str = "/n/netscratch/dam_lab/Lab/sqin/olmo/stage2/preprocessed",
                          tokenizer: str = "dolma2-tokenizer",
                          dtype=np.uint32,
                          use_remote: bool = False) -> None:
    """
    Build the complete data mix and write it to the YAML config file.

    Args:
        yaml_path: Path to the YAML config file to update
        base_dir: Base directory where preprocessed data is stored
        tokenizer: Tokenizer subdirectory name (default: 'dolma2-tokenizer')
        dtype: The numpy dtype of the memmap (default: np.uint32)
        use_remote: If True, write remote URLs; if False, write local paths (default: False)
    """
    print("Building data mix...")
    print()

    # Build the data mix
    shard_paths = build_data_mix(base_dir, tokenizer, dtype)

    print()
    print("Writing paths to YAML config...")

    # Write to YAML config
    write_paths_to_yaml(shard_paths, yaml_path, use_remote=use_remote)

    print()
    print(f"✓ Configuration complete!")
    print(f"  Config file: {yaml_path}")
    print(f"  Total shards: {len(shard_paths)}")


def write_gsm8k_sft_config(yaml_path: str = "/n/home05/sqin/OLMo/experiment_scripts/OLMo2-1B-sft.yaml",
                          base_dir: str = "/n/netscratch/dam_lab/Lab/sqin/olmo/stage2/preprocessed",
                          dtype=np.uint32,
                          num_epochs: int = 3,
                          global_batch_size: int = 128) -> None:
    """
    Build GSM8K SFT data mix and write to the SFT YAML config file.

    Args:
        yaml_path: Path to the SFT YAML config file to update
        base_dir: Base directory where preprocessed data is stored
        dtype: The numpy dtype of the memmap (default: np.uint32)
        num_epochs: Number of epochs to train (default: 3)
        global_batch_size: Global training batch size (default: 128)
    """
    print("=" * 80)
    print("CONFIGURING GSM8K SFT TRAINING")
    print("=" * 80)
    print()

    # Build the GSM8K data mix
    shard_paths = build_gsm8k_sft_mix(base_dir, dtype)

    # Calculate total tokens in dataset
    dataset_tokens = count_tokens(shard_paths, dtype)
    print(f"Dataset size: {dataset_tokens:,} tokens ({dataset_tokens/1e6:.2f}M)")
    print()

    # Calculate training parameters
    max_sequence_length = 4096
    tokens_per_step = global_batch_size * max_sequence_length
    steps_per_epoch = dataset_tokens / tokens_per_step
    total_steps = int(steps_per_epoch * num_epochs)
    total_training_tokens = dataset_tokens * num_epochs

    # Calculate warmup (5% of total training)
    warmup_tokens = int(total_training_tokens * 0.05)

    # Calculate evaluation and save intervals
    eval_interval = max(1, int(steps_per_epoch / 4))  # ~4 times per epoch
    save_interval = eval_interval
    save_interval_ephemeral = max(1, save_interval // 2)

    print("=" * 80)
    print("TRAINING CONFIGURATION:")
    print("=" * 80)
    print(f"Number of epochs:              {num_epochs}")
    print(f"Global batch size:             {global_batch_size}")
    print(f"Tokens per step:               {tokens_per_step:,}")
    print(f"Steps per epoch:               {steps_per_epoch:.1f}")
    print(f"Total training steps:          {total_steps}")
    print(f"Total training tokens:         {total_training_tokens:,} ({total_training_tokens/1e6:.2f}M)")
    print(f"Warmup tokens (5%):            {warmup_tokens:,} ({warmup_tokens/1e6:.2f}M)")
    print(f"Evaluation interval:           every {eval_interval} steps")
    print(f"Save interval:                 every {save_interval} steps")
    print("=" * 80)
    print()

    # Load the existing YAML config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update the configuration
    if 'data' not in config:
        config['data'] = {}

    config['data']['paths'] = shard_paths
    config['global_train_batch_size'] = global_batch_size
    config['max_duration'] = f"{total_training_tokens:.0f}T"
    config['stop_at'] = total_steps
    config['eval_interval'] = eval_interval
    config['save_interval'] = save_interval
    config['save_interval_ephemeral'] = save_interval_ephemeral

    if 'scheduler' not in config:
        config['scheduler'] = {}

    config['scheduler']['t_max'] = float(total_training_tokens)
    config['scheduler']['t_warmup'] = float(warmup_tokens)

    # Write back to the YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=1000)

    print("✓ Updated configuration file")
    print(f"  Config: {yaml_path}")
    print(f"  Data shards: {len(shard_paths)}")
    print(f"  Training: {num_epochs} epochs × {dataset_tokens/1e6:.2f}M tokens = {total_training_tokens/1e6:.2f}M total")
    print(f"  Steps: {total_steps} steps ({steps_per_epoch:.1f} steps/epoch)")
    print()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "build":
            # Build the complete data mix
            shards = build_data_mix()
            print(f"\nGenerated {len(shards)} total shard paths")
        elif sys.argv[1] == "write":
            # Build data mix and write to YAML config
            yaml_path = sys.argv[2] if len(sys.argv) > 2 else "/n/home05/sqin/OLMo/experiment_scripts/OLMo2-1B-stage1.yaml"
            build_and_write_config(yaml_path)
        else:
            # Count tokens for a specific source
            source_name = sys.argv[1]
            print(f"Source: {source_name}")

            shards = get_local_shards(source_name)
            print(f"Number of shards: {len(shards)}")

            total_tokens = count_tokens(shards)
            print(f"Total tokens: {total_tokens:,} ({total_tokens/1e6:.2f}M)")

            if len(shards) > 0:
                tokens_per_shard = count_tokens(shards[0])
                print(f"Tokens in first shard: {tokens_per_shard:,}")
    else:
        print("Usage:")
        print("  python data_utils.py <source_name>       - Count tokens for a source")
        print("  python data_utils.py build               - Build complete data mix")
        print("  python data_utils.py write [yaml_path]   - Build mix and write to YAML config")
        print()
        print("Examples:")
        print("  python data_utils.py personahub_math_v5_regen_149960")
        print("  python data_utils.py write")
        print("  python data_utils.py write /path/to/config.yaml")
