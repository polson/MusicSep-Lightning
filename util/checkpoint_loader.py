import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


@dataclass
class ParameterMatchResult:
    model_dict: Dict[str, torch.Tensor]
    matching_keys: List[str]
    checkpoint_only_keys: List[str]
    model_only_keys: List[str]


@dataclass
class LoadingSummary:
    matched_count: int
    total_checkpoint_params: int
    total_model_params: int
    unmatched_checkpoint: List[str]
    unmatched_model: List[str]
    checkpoint_info: Dict


class SmartCheckpointLoader:

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def load_checkpoint(self,
                        model: torch.nn.Module,
                        checkpoint_path: str) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_dict = checkpoint["state_dict"]
            model_dict = model.state_dict()
            result = self._match_params(checkpoint_dict, model_dict)
            model.load_state_dict(result.model_dict, strict=False)
            self._print_loading_summary(
                LoadingSummary(
                    matched_count=len(result.matching_keys),
                    total_checkpoint_params=len(checkpoint_dict),
                    total_model_params=len(model_dict),
                    unmatched_checkpoint=result.checkpoint_only_keys,
                    unmatched_model=result.model_only_keys,
                    checkpoint_info=self._extract_checkpoint_info(checkpoint)
                )
            )

        except Exception as e:
            logger.error(f"Error during smart checkpoint loading: {str(e)}")
            raise

    @staticmethod
    def _similarity(str1: str, str2: str) -> float:
        return SequenceMatcher(None, str1, str2).ratio()

    def _match_params(self,
                      checkpoint: Dict[str, torch.Tensor],
                      model: Dict[str, torch.Tensor]):
        checkpoint_keys = list(checkpoint.keys())
        model_keys = list(model.keys())

        similarity_matrix = []
        for model_key in model_keys:
            for checkpoint_key in checkpoint_keys:
                similarity = self._similarity(model_key, checkpoint_key)
                if similarity > self.similarity_threshold:
                    similarity_matrix.append((model_key, checkpoint_key, similarity))

        best_matches = {}
        used_checkpoint_keys = set()
        sorted_matches = sorted(similarity_matrix, key=lambda x: x[2], reverse=True)
        for model_key, checkpoint_key, similarity in sorted_matches:
            if model_key not in best_matches and checkpoint_key not in used_checkpoint_keys:
                model_tensor = model[model_key]
                checkpoint_tensor = checkpoint[checkpoint_key]
                if model_tensor.shape == checkpoint_tensor.shape:
                    best_matches[model_key] = (model_key, checkpoint_key, similarity)
                    used_checkpoint_keys.add(checkpoint_key)

        result_dict = model.copy()
        matched_model_keys = set()
        matched_checkpoint_keys = set()
        for model_key, (_, checkpoint_key, _) in best_matches.items():
            result_dict[model_key] = checkpoint[checkpoint_key]
            matched_model_keys.add(model_key)
            matched_checkpoint_keys.add(checkpoint_key)

        matching_keys = list(matched_model_keys)
        checkpoint_only_keys = [key for key in checkpoint_keys if key not in matched_checkpoint_keys]
        model_only_keys = [key for key in model_keys if key not in matched_model_keys]

        return ParameterMatchResult(
            model_dict=result_dict,
            matching_keys=matching_keys,
            checkpoint_only_keys=checkpoint_only_keys,
            model_only_keys=model_only_keys
        )

    def _extract_checkpoint_info(self, checkpoint: Dict) -> Dict:
        keys_to_extract = ['epoch', 'global_step', 'pytorch-lightning_version', 'lr_schedulers', 'optimizer_states',
                           'hyper_parameters']
        return {key: checkpoint[key] for key in keys_to_extract if key in checkpoint}

    def _print_loading_summary(self, summary: LoadingSummary):
        print("\n" + "=" * 60)
        print("SMART CHECKPOINT LOADING SUMMARY")
        print("=" * 60)
        print(f"✓ Matched tensors: {summary.matched_count}/{summary.total_model_params}")

        if summary.unmatched_model:
            print(f"⚠ Unmatched model tensors ({len(summary.unmatched_model)}):")
            for name in summary.unmatched_model[:5]:
                print(f"    - {name}")
            if len(summary.unmatched_model) > 5:
                print(f"    ... and {len(summary.unmatched_model) - 5} more")

        if summary.unmatched_checkpoint:
            print(f"⚠ Unmatched checkpoint tensors ({len(summary.unmatched_checkpoint)}):")
            for name in summary.unmatched_checkpoint[:5]:
                print(f"    - {name}")
            if len(summary.unmatched_checkpoint) > 5:
                print(f"    ... and {len(summary.unmatched_checkpoint) - 5} more")

        if 'epoch' in summary.checkpoint_info:
            print(f"📊 Checkpoint epoch: {summary.checkpoint_info['epoch']}")
        if 'global_step' in summary.checkpoint_info:
            print(f"📊 Global step: {summary.checkpoint_info['global_step']}")
        print("=" * 60)
