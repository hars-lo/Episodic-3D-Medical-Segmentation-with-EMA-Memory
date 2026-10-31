from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch


def _ensure_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")
    if tensor.ndim < 2:
        raise ValueError(
            "Task embeddings must have shape (..., m+1, C). "
            f"Received tensor with ndim={tensor.ndim}"
        )
    return tensor


@dataclass
class ClassMemoryBank:
    momentum: float = 0.9 
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    storage: MutableMapping[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("Momentum α must lie in [0, 1).")
    def __contains__(self, class_id: int) -> bool:
        return int(class_id) in self.storage

    def get(self, class_id: int) -> Optional[torch.Tensor]:
        tensor = self.storage.get(int(class_id))
        if tensor is None:
            return None
        return tensor.detach().clone()

    def items(self) -> Iterable[tuple[int, torch.Tensor]]:
        for key, tensor in self.storage.items():
            yield key, tensor.detach().clone()

    def clear(self) -> None:
        self.storage.clear()
    def update(self, class_id: int, embedding: torch.Tensor) -> torch.Tensor:
        class_id = int(class_id)
        embedding = _ensure_tensor(embedding).detach()
        if embedding.ndim != 2:
            raise ValueError(
                "Embedding must have rank 2 with shape (m+1, C); "
                f"received {tuple(embedding.shape)}"
            )

        if self.device is None:
            self.device = embedding.device
        if self.dtype is None:
            self.dtype = embedding.dtype

        embedding = embedding.to(device=self.device, dtype=self.dtype)

        if class_id in self.storage:
            stored = self.storage[class_id]
            if stored.shape != embedding.shape:
                raise ValueError(
                    f"Shape mismatch for class {class_id}: "
                    f"existing {tuple(stored.shape)} vs new {tuple(embedding.shape)}"
                )
            updated = self.momentum * stored + (1.0 - self.momentum) * embedding
        else:
            updated = embedding

        self.storage[class_id] = updated
        return updated.detach().clone()

    def update_episode(
        self,
        task_embeddings: torch.Tensor,
        class_ids: Sequence[Sequence[int]] | Sequence[int],
    ) -> None:
        embeddings = _ensure_tensor(task_embeddings).detach()
        if embeddings.ndim == 3: 
            embeddings = embeddings.unsqueeze(0)

        if isinstance(class_ids[0], (list, tuple)):
            flat_class_ids: List[List[int]] = [
                [int(cid) for cid in sample_ids] for sample_ids in class_ids 
            ]
        else:
            flat_class_ids = [list(map(int, class_ids))]  

        if embeddings.shape[0] != len(flat_class_ids):
            raise ValueError(
                "Mismatch between batch size of task embeddings and class_ids."
            )

        for sample_embeddings, sample_classes in zip(embeddings, flat_class_ids):
            if len(sample_classes) != sample_embeddings.shape[0]:
                raise ValueError(
                    "Number of class IDs per sample must match embedding count."
                )
            for class_id, class_embedding in zip(sample_classes, sample_embeddings):
                if int(class_id) == 0:
                    continue  
                self.update(int(class_id), class_embedding)

    def retrieve(
        self,
        class_ids: Sequence[int],
        *,
        default: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        retrieved: List[torch.Tensor] = []
        for class_id in class_ids:
            tensor = self.storage.get(int(class_id))
            if tensor is None:
                if default is None:
                    raise KeyError(
                        f"Class {class_id} is not present in the memory bank."
                    )
                tensor = default
            retrieved.append(tensor.detach())

        batch = torch.stack(retrieved, dim=0)
        if device is not None or dtype is not None:
            batch = batch.to(device=device or batch.device, dtype=dtype or batch.dtype)
        return batch

    @staticmethod
    def ensemble(embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
        if not embeddings:
            raise ValueError("Cannot build ensemble from an empty sequence.")

        stacked = torch.stack([_ensure_tensor(t).detach() for t in embeddings], dim=0)
        return stacked.mean(dim=0)

    def summary(self) -> Mapping[int, torch.Size]:
        return {class_id: tensor.shape for class_id, tensor in self.storage.items()}



