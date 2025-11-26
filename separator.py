from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from einops import rearrange


class Separator:
    def __init__(
            self,
            target_sources,
            batch_size=4,
            overlap_percent=0.5,
            chunk_size_seconds=10,
            steps=10  # <--- Added: Number of inference steps
    ):
        super().__init__()
        self.target_sources = target_sources
        self.batch_size = batch_size
        self.chunk_size = chunk_size_seconds * 44100
        self.overlap_size = round(self.chunk_size * overlap_percent)
        self.steps = steps  # <--- Store steps

    def process_file(
            self,
            model,
            mixture_path: str,
            output_dir: str = None
    ):
        waveform, sample_rate = torchaudio.load(mixture_path)
        chunks = self._create_chunks(waveform)

        ext = Path(mixture_path).suffix.lower()
        processed_chunks = []
        model.eval()

        for chunk in chunks:
            chunk = chunk.to(model.device)
            with torch.no_grad():
                output = model.iterative_inference(chunk, steps=self.steps)
            processed_chunks.append(output)

        recombined = self._overlap_add_chunks(
            processed_chunks,
            waveform.shape[-1],
        )

        stems = torch.unbind(recombined, dim=0)

        output_dir = Path(output_dir) if output_dir else None
        if output_dir:
            input_parent_folder = Path(mixture_path).parent.name
            file_output_dir = output_dir / input_parent_folder
            file_output_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for i, stem in enumerate(stems):
                output_name = Path(mixture_path).stem
                stem = stem.to("cpu")
                path = file_output_dir / f"{output_name}_{self.target_sources[i]}{ext}"
                torchaudio.save(path, stem, sample_rate)
                count = count + 1

        return stems

    def _create_chunks(
            self,
            waveform: torch.Tensor
    ):
        step_size = self.chunk_size - self.overlap_size
        waveform = F.pad(waveform, (step_size, step_size))
        chunked = waveform.unfold(dimension=-1, size=self.chunk_size, step=step_size)
        chunked = rearrange(chunked, "c n t -> n c t")

        n, c, t = chunked.shape
        num_tensors = round(n / self.batch_size)

        # Handle case where num_tensors is 0 or very small
        if num_tensors == 0:
            num_tensors = 1

        tensors = torch.chunk(chunked, num_tensors, dim=0)
        return tensors

    def _overlap_add_chunks(
            self,
            chunks: list[torch.Tensor],
            original_waveform_length: int,
    ) -> torch.Tensor:
        step_size = self.chunk_size - self.overlap_size
        combined_chunks = torch.cat(chunks, dim=0)

        padded_output_length = original_waveform_length + 2 * step_size

        b, n, c, t = combined_chunks.shape
        output_waveform = torch.zeros(
            n, c, padded_output_length,
            device=combined_chunks.device
        )

        hann_window = torch.hann_window(self.chunk_size, device=combined_chunks.device)

        for i in range(b):
            start_idx = i * step_size
            end_idx = start_idx + self.chunk_size

            windowed_chunk = combined_chunks[i] * hann_window
            output_waveform[:, :, start_idx:end_idx] += windowed_chunk

        restored_waveform = output_waveform[:, :, step_size:-step_size]

        return restored_waveform
