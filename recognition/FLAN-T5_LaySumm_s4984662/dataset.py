"""
dataset.py - data loading and preprocessing for BioLaySumm 2025 LaymanRRG (open-source track).

Author : Truong Trung Bao (s4984662)
Course : COMP3710 Pattern Recognition, UQ, Semester 2 2026 - Project 2.6
         (FLAN-T5 lay summarisation of radiology reports, LoRA vs full fine-tuning)

Description
    Turns (radiology report, lay summary) pairs into the tensors FLAN-T5 needs:
      1. a task prompt is prepended to the report (FLAN-T5 is instruction-tuned);
      2. the prompted report and the lay summary are tokenised and truncated;
      3. when a batch is padded, padded *label* positions are set to -100 so the loss
         ignores them (target sequence masking).

Key components
    load_laysumm       - load the three official splits from the Hugging Face Hub
    build_source_text  - prepend the task prompt to a radiology report
    LaySummDataset     - tokenise one example (truncation) as lists of token ids
    make_collate_fn    - pad a batch; padded labels become -100

Self-check (needs `transformers`, run on Rangpur; downloads nothing if the cache is warm):
    python dataset.py            # synthetic checks
    python dataset.py --real     # also loads LaymanRRG and tokenises 1,000 validation rows

References
    BioLaySumm 2025 LaymanRRG: https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track
    T5 (Raffel et al., 2020) and FLAN-T5 (Chung et al., 2024), spec references [19], [20].
    Hugging Face T5 documentation: label positions equal to -100 are ignored by the loss.

AI assistance: drafted with Claude (Anthropic) and reviewed by the author, see the README
section "Artificial Intelligence Usage Disclosure".
"""

import argparse
from typing import Dict, List, Optional

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset

DATASET_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
SOURCE_COLUMN = "radiology_report"
TARGET_COLUMN = "layman_report"

# Truncation lengths come from the data audit (feasibility review, section 3): with the
# FLAN-T5 tokenizer only 0.19% of train reports and 0.24% of train summaries exceed 512
# tokens, so truncating at 512 affects at most ~0.4% of rows.
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512

# PyTorch's CrossEntropyLoss (and Hugging Face T5's loss) skips targets equal to -100.
IGNORE_INDEX = -100

# The prompt goes BEFORE the report so truncation can only cut the end of the report,
# never the instruction.
TASK_PROMPT = "Rewrite the following radiology report in plain language that a patient can understand:"


def load_laysumm(cache_dir: Optional[str] = None) -> DatasetDict:
    """Load the official train / validation / test splits (no re-splitting, no shuffling).

    The test split has an empty `layman_report` for every row (blind shared-task test
    set), so only train and validation can be used to compute ROUGE.
    """
    return load_dataset(DATASET_ID, cache_dir=cache_dir)


def build_source_text(report: str, prompt: str = TASK_PROMPT) -> str:
    """Return the model input: the task prompt followed by the radiology report."""
    return f"{prompt} {report.strip()}"


class LaySummDataset(Dataset):
    """One split of LaymanRRG as tokenised (input_ids, attention_mask[, labels]) examples.

    Tokenisation is done on the fly in `__getitem__` so no tokenised copy of the 150k
    training pairs is kept in memory. Items are plain lists of ints; padding to a common
    length is done per batch by `make_collate_fn`.

    Args:
        split: a Hugging Face split (or any mapping of column name -> list of str).
        tokenizer: the FLAN-T5 tokenizer.
        prompt: task prompt prepended to every report.
        max_source_len / max_target_len: truncation lengths (in tokens).
        with_targets: False for the test split, which has no reference summaries.
    """

    def __init__(self, split, tokenizer, prompt: str = TASK_PROMPT,
                 max_source_len: int = MAX_SOURCE_LEN, max_target_len: int = MAX_TARGET_LEN,
                 with_targets: bool = True) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        # list(...) materialises the column: datasets>=5 returns lazy Column objects.
        self.reports: List[str] = [r or "" for r in list(split[SOURCE_COLUMN])]
        self.targets: Optional[List[str]] = None
        if with_targets:
            self.targets = [t or "" for t in list(split[TARGET_COLUMN])]
            if not any(t.strip() for t in self.targets):
                raise ValueError("every reference summary is empty (this is the blind test "
                                 "split); build the dataset with with_targets=False")

    def __len__(self) -> int:
        return len(self.reports)

    def __getitem__(self, index: int) -> Dict[str, object]:
        source = build_source_text(self.reports[index], self.prompt)
        # truncation=True cuts the text but the tokenizer still appends </s> afterwards.
        encoded = self.tokenizer(source, max_length=self.max_source_len, truncation=True)
        item: Dict[str, object] = {"input_ids": encoded["input_ids"],
                                   "attention_mask": encoded["attention_mask"],
                                   "index": index}
        if self.targets is not None:
            # The </s> token stays in the labels (it is NOT masked): the model must learn
            # to emit it, otherwise generation would never stop on its own.
            target = self.tokenizer(self.targets[index], max_length=self.max_target_len,
                                    truncation=True)
            item["labels"] = target["input_ids"]
        return item


def make_collate_fn(pad_token_id: int):
    """Build a DataLoader `collate_fn` that pads each batch to its longest sequence.

    Inputs are padded with the tokenizer's pad id (and attention_mask 0). Labels are
    padded with -100 instead of the pad id, so padded positions contribute nothing to the
    loss. Hugging Face T5 then derives the decoder inputs by shifting the labels right
    and replacing -100 with the pad id, so no extra step is needed.
    """

    def pad(sequences: List[List[int]], value: int) -> torch.Tensor:
        width = max(len(seq) for seq in sequences)
        return torch.tensor([seq + [value] * (width - len(seq)) for seq in sequences],
                            dtype=torch.long)

    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        out = {"input_ids": pad([b["input_ids"] for b in batch], pad_token_id),
               "attention_mask": pad([b["attention_mask"] for b in batch], 0),
               "index": torch.tensor([b["index"] for b in batch], dtype=torch.long)}
        if "labels" in batch[0]:
            out["labels"] = pad([b["labels"] for b in batch], IGNORE_INDEX)
        return out

    return collate


# --------------------------------------------------------------------------------------
# Self-check. Run on Rangpur; it only needs the tokenizer (a few MB, no model weights).
# --------------------------------------------------------------------------------------
def _self_check(model_name: str, real: bool) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    prompt_ids = tokenizer(TASK_PROMPT)["input_ids"][:-1]  # drop </s>

    split = {SOURCE_COLUMN: ["No pneumothorax.", "Mild bibasilar atelectasis. " * 4,
                             "pneumothorax " * 3000],
             TARGET_COLUMN: ["There is no collapsed lung.", "Small areas at the bottom of "
                             "both lungs are not fully open.", "x"]}
    dataset = LaySummDataset(split, tokenizer)
    items = [dataset[i] for i in range(len(dataset))]

    for item in items:  # prompt first, length cap, </s> kept
        assert item["input_ids"][:len(prompt_ids)] == prompt_ids, "prompt must come first"
        assert len(item["input_ids"]) <= MAX_SOURCE_LEN and item["input_ids"][-1] == eos
        assert item["labels"][-1] == eos
    assert len(items[2]["input_ids"]) == MAX_SOURCE_LEN, "long report must be truncated"

    batch = make_collate_fn(pad)(items)
    padding = batch["attention_mask"] == 0
    assert (batch["input_ids"][padding] == pad).all(), "inputs padded with the pad id"
    label_padding = batch["labels"] == IGNORE_INDEX
    expected = sum(batch["labels"].shape[1] - len(it["labels"]) for it in items)
    assert int(label_padding.sum()) == expected, "only padded label positions are -100"
    assert not (batch["labels"] == pad).any(), "no pad id may leak into labels"

    assert "labels" not in LaySummDataset(split, tokenizer, with_targets=False)[0]
    try:
        LaySummDataset({SOURCE_COLUMN: ["a"], TARGET_COLUMN: [""]}, tokenizer)
    except ValueError:
        pass
    else:
        raise AssertionError("an all-empty reference column must be rejected")
    print("synthetic checks passed")

    if real:
        splits = load_laysumm()
        print({name: len(part) for name, part in splits.items()})
        validation = LaySummDataset(splits["validation"].select(range(1000)), tokenizer)
        truncated = sum(len(validation[i]["input_ids"]) == MAX_SOURCE_LEN for i in range(1000))
        print(f"validation rows truncated at {MAX_SOURCE_LEN} tokens (first 1,000): {truncated}")
        test = LaySummDataset(splits["test"].select(range(8)), tokenizer, with_targets=False)
        assert "labels" not in test[0]
        print("real-data checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-check for dataset.py")
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--real", action="store_true", help="also load LaymanRRG")
    args = parser.parse_args()
    _self_check(args.model_name, args.real)
