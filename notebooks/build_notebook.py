"""Builds finetune_qwen_lora.ipynb from the cells below, so the notebook stays reviewable as code.
Run: python notebooks/build_notebook.py
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# Fine-tuning Qwen2.5-1.5B with LoRA to classify customer messages

Runs on a free T4 GPU in about 20 minutes, on Colab or Kaggle.

- **Colab:** Runtime → Change runtime type → T4 GPU, then Runtime → Run all.
- **Kaggle:** in the notebook's settings panel, Accelerator → GPU T4 x2 and Internet → on (needs a verified phone), then Run all. The result is saved in the Output panel.

1. Load the data from the project's GitHub repository.
2. Measure the model **before** fine-tuning, on the hand-written test set.
3. Fine-tune with LoRA on the synthetic training set.
4. Measure again, on the same test set.
5. Download the predictions (`lora-predictions.json`) and, optionally, publish the adapter to Hugging Face.
""")

code("""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Kaggle gives two T4s; one is enough and avoids multi-GPU surprises
!pip install -q -U peft  # Colab ships an older peft that does not match its newer transformers
!pip uninstall -q -y torchao  # unused here; new peft refuses the old version Colab ships
!rm -rf repo && git clone -q https://github.com/Flaviohorbach/intent-classifier-finetune.git repo
import json, time, torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none: switch the runtime to T4 GPU")
""")

code("""
def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

train = read_jsonl("repo/data/train.jsonl")
val = read_jsonl("repo/data/val.jsonl")
test = read_jsonl("repo/data/test.jsonl")
LABELS = ["troca", "atraso", "duvida", "reclamacao_grave", "outro"]
print(len(train), "train |", len(val), "val |", len(test), "test")
""")

md("""
## The prompt

Short on purpose: the fine-tuned model learns the labels from examples, not from a long guide. The same prompt is used before and after fine-tuning.
""")

code("""
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"  # generation pads on the left
# Newer transformers call it `dtype`; older ones, `torch_dtype`.
from packaging.version import Version
dtype_arg = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda", **{dtype_arg: torch.float16})

def prompt(text):
    return (
        "Classifique a mensagem de um cliente de loja online em uma destas categorias: "
        "troca, atraso, duvida, reclamacao_grave, outro.\\n"
        f"Mensagem: {text}\\nCategoria:"
    )

def parse(output):
    word = output.strip().lower().split()[0] if output.strip() else ""
    word = word.strip(".,:;!\\"'")
    return next((label for label in LABELS if word.startswith(label[:5])), "invalido")

@torch.no_grad()
def classify(texts, batch_size=16):
    model.eval()
    results = []
    for i in range(0, len(texts), batch_size):
        batch = [prompt(t) for t in texts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        decoded = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        results += [parse(d) for d in decoded]
    return results

@torch.no_grad()
def latency_ms(texts):
    # One message at a time, as a live classifier would see them.
    times = []
    for t in texts:
        start = time.perf_counter()
        classify([t], batch_size=1)
        times.append((time.perf_counter() - start) * 1000)
    return sorted(times)[len(times) // 2]

def report(name, predicted):
    right = sum(p == r["label"] for p, r in zip(predicted, test))
    invalid = predicted.count("invalido")
    print(f"{name}: {right}/{len(test)} right ({right / len(test):.1%}), {invalid} answers that are not a label")
""")

md("## 1. Before fine-tuning")

code("""
before = classify([r["text"] for r in test])
report("Qwen2.5-1.5B, prompt only", before)
before_ms = latency_ms([r["text"] for r in test[:20]])
print(f"median latency, one message at a time: {before_ms:.0f} ms")
""")

md("""
## 2. Fine-tuning with LoRA

LoRA trains two small matrices next to each attention and MLP layer instead of all 1.5 billion weights: about 1% of the parameters. The loss counts only the label tokens, not the prompt, so the model learns to answer and not to repeat messages.
""")

code("""
from peft import LoraConfig, get_peft_model

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
# The T4 has no bfloat16: train the small LoRA weights in float32 next to the float16 model.
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()

def encode(row):
    prompt_ids = tok(prompt(row["text"]))["input_ids"]
    label_ids = tok(" " + row["label"])["input_ids"] + [tok.eos_token_id]
    return {"input_ids": prompt_ids + label_ids, "labels": [-100] * len(prompt_ids) + label_ids}

def collate(rows):
    width = max(len(r["input_ids"]) for r in rows)
    pad = lambda seq, value: [value] * (width - len(seq)) + seq
    return {
        "input_ids": torch.tensor([pad(r["input_ids"], tok.pad_token_id) for r in rows]),
        "attention_mask": torch.tensor([pad([1] * len(r["input_ids"]), 0) for r in rows]),
        "labels": torch.tensor([pad(r["labels"], -100) for r in rows]),
    }

train_set = [encode(r) for r in train]
val_set = [encode(r) for r in val]
""")

code("""
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="out", num_train_epochs=2, per_device_train_batch_size=16, per_device_eval_batch_size=16,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05, fp16=True,
    logging_steps=20, eval_strategy="epoch", save_strategy="no", report_to="none", remove_unused_columns=False,
)
trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=val_set, data_collator=collate)
start = time.time()
trainer.train()
train_minutes = (time.time() - start) / 60
print(f"trained in {train_minutes:.1f} min")
""")

md("## 3. After fine-tuning")

code("""
after = classify([r["text"] for r in test])
report("Qwen2.5-1.5B + LoRA", after)
after_ms = latency_ms([r["text"] for r in test[:20]])
print(f"median latency, one message at a time: {after_ms:.0f} ms")
""")

md("## 4. Download the predictions")

code("""
result = {
    "model": MODEL,
    "gpu": torch.cuda.get_device_name(0),
    "trainMinutes": round(train_minutes, 1),
    "trainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "before": {"latencyMs": round(before_ms), "predictions": [{"text": r["text"], "label": r["label"], "predicted": p} for r, p in zip(test, before)]},
    "after": {"latencyMs": round(after_ms), "predictions": [{"text": r["text"], "label": r["label"], "predicted": p} for r, p in zip(test, after)]},
}
json.dump(result, open("lora-predictions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
try:
    from google.colab import files  # Colab: the browser downloads it
    files.download("lora-predictions.json")
except ImportError:  # Kaggle: it appears in the Output panel
    print("saved to", os.path.abspath("lora-predictions.json"))
""")

md("""
## 5. Optional: publish the adapter to Hugging Face

Add a Hugging Face token with **write** access as a secret named `HF_TOKEN` (Colab: key icon on the left; Kaggle: Add-ons → Secrets), allow this notebook to read it, and run the cell. Only the LoRA adapter is uploaded (tens of MB), not the base model.
""")

code("""
from huggingface_hub import login
try:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
except ImportError:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
login(token)
REPO = "intent-classifier-qwen2.5-1.5b-lora"
model.push_to_hub(REPO, private=False)
tok.push_to_hub(REPO, private=False)
print("published")
""")

notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 0,
}
Path(__file__).with_name("finetune_qwen_lora.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("written")
