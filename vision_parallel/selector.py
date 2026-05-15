from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import torch

class AttentionSelector(ABC):
    
    @abstractmethod
    def analyze(self, attention_scores: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        pass
    
    @abstractmethod
    def select_kv_cache(self, kv_cache: Dict[str, torch.Tensor], important_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass