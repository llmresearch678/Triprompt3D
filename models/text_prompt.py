"""
text_prompt.py
Text Prompt Encoder (Qt) for TRIPROMPT-3D.
Uses ClinicalBERT (BERT-Base fine-tuned on MIMIC-III) to map
per-class anatomical descriptions to semantic embeddings Qt.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# Default class descriptions used in the paper
DEFAULT_ORGAN_DESCRIPTIONS = {
    "liver":        "The liver is the largest solid organ in the abdominal cavity, "
                    "with a smooth homogeneous parenchyma and well-defined capsule.",
    "kidney_l":     "The left kidney is a bean-shaped retroperitoneal organ with a "
                    "distinct cortex and medulla, located in the left flank.",
    "kidney_r":     "The right kidney is a bean-shaped retroperitoneal organ, "
                    "slightly lower than the left, with a cortex and medullary pyramids.",
    "spleen":       "The spleen is a vascular abdominal organ with homogeneous "
                    "parenchyma, located in the left upper quadrant.",
    "pancreas":     "The pancreas is a retroperitoneal gland with a head, body, and "
                    "tail, often irregular in shape and variable in size.",
    "gallbladder":  "The gallbladder is a small pear-shaped sac beneath the liver "
                    "that stores bile, highly variable in size and position.",
    "stomach":      "The stomach is a hollow muscular organ in the upper abdomen "
                    "with a curved body, fundus, and antrum.",
    "aorta":        "The aorta is the main arterial trunk descending through the "
                    "thorax and abdomen with a circular lumen.",
    "duodenum":     "The duodenum is the first and shortest segment of the small "
                    "intestine, forming a C-shaped curve around the pancreatic head.",
    "tumor":        "A tumor is an abnormal mass of tissue exhibiting irregular "
                    "borders, heterogeneous density, and variable deformation of "
                    "adjacent anatomical structures.",
}


class TextPromptEncoder(nn.Module):
    """
    Encodes per-class anatomical text descriptions into semantic
    prompt embeddings Qt using a pretrained language model.

    The encoder is Lipschitz-continuous w.r.t. embedding norm,
    ensuring bounded perturbation sensitivity.

    Args:
        model_name  : HuggingFace model name (ClinicalBERT by default)
        feature_dim : shared backbone dim C — output projection size
        freeze_bert : freeze BERT weights (recommended for fine-tuning)
        max_length  : maximum token length for text inputs
    """

    CLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"

    def __init__(
        self,
        model_name=None,
        feature_dim=256,
        freeze_bert=True,
        max_length=128,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_length  = max_length

        model_name = model_name or self.CLINICAL_BERT

        # Load tokenizer and BERT encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert      = AutoModel.from_pretrained(model_name)

        bert_dim = self.bert.config.hidden_size   # 768 for BERT-Base

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # Learnable projection Wt: R^{Ct} → R^C
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # Cache for encoded prompts (avoid re-encoding every forward pass)
        self._cache = {}

    @torch.no_grad()
    def encode_texts(self, texts, device):
        """
        Tokenize and encode a list of text strings.

        Args:
            texts : list of str, length K
            device: target device

        Returns:
            embeddings: (K, bert_dim) CLS token embeddings
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        outputs = self.bert(**tokens)
        # Use [CLS] token as sentence representation
        cls_emb = outputs.last_hidden_state[:, 0, :]   # (K, bert_dim)
        return cls_emb

    def forward(self, class_descriptions, device=None):
        """
        Encode class text descriptions into prompt matrix Qt.

        Args:
            class_descriptions: dict {class_name: text_str} or list of str
            device            : target device (inferred if None)

        Returns:
            Qt: (1, K, C)  text prompt matrix, broadcastable over batch
        """
        if isinstance(class_descriptions, dict):
            class_names = list(class_descriptions.keys())
            texts       = list(class_descriptions.values())
        else:
            texts       = class_descriptions
            class_names = [str(i) for i in range(len(texts))]

        device = device or next(self.proj.parameters()).device

        # Check cache
        cache_key = tuple(class_names)
        if cache_key not in self._cache:
            raw_emb = self.encode_texts(texts, device)   # (K, bert_dim)
            self._cache[cache_key] = raw_emb.to(device)

        raw_emb = self._cache[cache_key].to(device)      # (K, bert_dim)
        Qt = self.proj(raw_emb)                           # (K, C)
        Qt = Qt.unsqueeze(0)                              # (1, K, C)
        return Qt

    def clear_cache(self):
        self._cache.clear()
