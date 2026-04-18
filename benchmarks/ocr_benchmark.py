"""
OCR benchmark — vision-language model text extraction accuracy.

Dataset format expected in benchmarks/data/ocr/:
  Each sample is a directory containing:
    - image.png  (the document/image to OCR)
    - ground_truth.txt  (expected extracted text)

Score: Character Error Rate (CER). Lower is better.
Score is stored as (1 - CER) so higher = better, consistent with other benchmarks.

To use a real dataset, populate benchmarks/data/ocr/ with your samples
or override load_dataset() to point at your data source.
"""
from __future__ import annotations
import base64
from pathlib import Path
from benchmarks.base import BaseBenchmark

DATA_DIR = Path(__file__).parent / "data" / "ocr"


def _cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate via simple edit distance."""
    h, r = list(hypothesis), list(reference)
    if not r:
        return 0.0 if not h else 1.0
    # DP edit distance
    d = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(len(h) + 1):
        d[i][0] = i
    for j in range(len(r) + 1):
        d[0][j] = j
    for i in range(1, len(h) + 1):
        for j in range(1, len(r) + 1):
            cost = 0 if h[i - 1] == r[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(h)][len(r)] / len(r)


class OCRBenchmark(BaseBenchmark):
    name = "ocr"
    score_label = "OCR Accuracy (1-CER)"

    def load_dataset(self) -> list[dict]:
        """
        Load samples from DATA_DIR.
        Returns empty list if no data found — add your samples there.
        """
        samples = []
        if not DATA_DIR.exists():
            return samples
        for sample_dir in sorted(DATA_DIR.iterdir()):
            img = sample_dir / "image.png"
            gt = sample_dir / "ground_truth.txt"
            if img.exists() and gt.exists():
                samples.append({
                    "id": sample_dir.name,
                    "image_path": str(img),
                    "reference": gt.read_text().strip(),
                })
        return samples

    def build_prompt(self, sample: dict) -> str:
        """
        For VLM backends: encode image as base64 and embed in prompt.
        Note: actual multimodal prompting format varies by backend/model.
        This returns a text description for text-only fallback.
        Override for your specific VLM prompt format.
        """
        img_path = Path(sample["image_path"])
        if img_path.exists():
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            # Placeholder — adapt this to your model's multimodal format
            return f"[IMAGE:{b64}]\nExtract all text from this image. Return only the extracted text."
        return "Extract all text from the provided image."

    def score_sample(self, prediction: str, reference: str) -> float:
        cer = _cer(prediction.strip(), reference.strip())
        return max(0.0, 1.0 - cer)  # Convert to accuracy-style: higher = better
