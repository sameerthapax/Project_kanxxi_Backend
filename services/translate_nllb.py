# services/translate_nllb.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass(frozen=True)
class TranslationResult:
    text: str
    model_id: str
    src_lang: str
    tgt_lang: str


class NLLBTranslator:
    """
    English -> Nepali translator using NLLB-200.
    Loads model once and reuses it for all calls.
    """

    def __init__(
            self,
            model_id: str = "facebook/nllb-200-distilled-600M",
            device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device or self._pick_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        self.model.eval()

        # FLORES-200 language codes
        self.SRC = "eng_Latn"
        self.TGT = "npi_Deva"

        # cache token id lookup
        self._forced_bos_id = self.tokenizer.convert_tokens_to_ids(self.TGT)

    def _pick_device(self) -> str:
        # For Mac: prefer MPS if available
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @torch.inference_mode()
    def translate_en_to_ne(
            self,
            text: str,
            *,
            max_new_tokens: int = 256,
            num_beams: int = 4,
    ) -> TranslationResult:
        text = (text or "").strip()
        if not text:
            return TranslationResult(text="", model_id=self.model_id, src_lang=self.SRC, tgt_lang=self.TGT)

        # IMPORTANT: set src lang each call (tokenizer is stateful)
        self.tokenizer.src_lang = self.SRC

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        out = self.model.generate(
            **inputs,
            forced_bos_token_id=self._forced_bos_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        ne = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return TranslationResult(text=ne, model_id=self.model_id, src_lang=self.SRC, tgt_lang=self.TGT)


# Singleton accessor (lazy init)
@lru_cache(maxsize=1)
def get_translator() -> NLLBTranslator:
    return NLLBTranslator()
if __name__ == "__main__":
    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(description="CLI: English -> Nepali using NLLB-200")
    parser.add_argument("text", nargs="*", help="English text to translate. If empty, reads from stdin.")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M", help="HF model id")
    parser.add_argument("--beams", type=int, default=4, help="Beam size (quality vs speed)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens")
    args = parser.parse_args()

    # Get input
    if args.text:
        text = " ".join(args.text).strip()
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("No input text provided.", file=sys.stderr)
        sys.exit(1)

    # Build translator (explicit init so you can override model/device)
    t0 = time.time()
    translator = NLLBTranslator(model_id=args.model, device=args.device)
    t1 = time.time()

    result = translator.translate_en_to_ne(
        text,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.beams,
    )
    t2 = time.time()

    print(result.text)
    print(f"\n[info] model={result.model_id} device={translator.device} load={t1-t0:.2f}s translate={t2-t1:.2f}s", file=sys.stderr)