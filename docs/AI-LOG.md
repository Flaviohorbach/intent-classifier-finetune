# AI log: what the agent got wrong

Built with a coding agent (Claude Code), as in [project 5](https://github.com/Flaviohorbach/legacy-storefront-refactor). These happened while building this project.

### 1. The evaluation measured a slightly different model than the page runs

Training embedded messages in batches of 32; the page embeds one at a time. The 8-bit model scales activations per batch, so the same text produced a slightly different vector (cosine 0.995 instead of 1). "Bom dia" came out as a question in the evaluation and as "other" on the page. **Caught by:** typing 20 test messages into the page and comparing with the saved predictions, then embedding one message alone and inside a batch. **Fix:** training embeds one text at a time. The score went from 83.3% to 82.5%, and the page and evaluation now agree on 118 of 120.

### 2. A notebook written for a machine it could not test on

The Colab notebook failed three times in a row on the user's runs: a renamed argument (`torch_dtype` → `dtype`) in some versions, Colab's old `peft` refusing its newer `transformers`, and the new `peft` refusing Colab's old `torchao`. Each fix came from the error the user pasted. **Caught by:** the user running it. **Fix:** version-aware loading, upgrading only `peft`, and removing the unused `torchao`. A notebook pinned to tested versions would have avoided the round trips, at the cost of fighting Colab's own preinstalled packages.

### 3. An upgrade that may have broken the GPU

One fix added `pip install -U peft accelerate`, and the next run said PyTorch had no CUDA. The upgrade was the suspect, so it was removed, but the runtime may simply have reconnected without a GPU: the cause was not proven. **Caught by:** the user's next run. **Fix:** upgrade only what is needed.

### 4. A wrong number in a progress message

Reporting the keyword baseline, the agent compared 52.5% here with "77% on project 9". That 77% was project 9's routing score; on the same task, picking the label, keywords got 20 of 30 there (67%). **Caught by:** rereading project 9's results while writing the README. **Fix:** the README uses 20 of 30.

### 5. Label noise noticed by reading, not by a metric

The generated data looked fine in counts and balance. Reading a random sample showed a message about a cut hand labeled as an exchange, and counting found that serious complaints were mostly about money (72) and rarely about health (17). **Caught by:** reading samples before training. **Fix:** none in the data, on purpose: correcting it after seeing the test results would bias them. The README explains the gap instead.

## What needed a human

- Choosing both a browser model and a LoRA model, and synthetic data with a hand-written test.
- Running the notebook on Colab, which needs a Google account and a free GPU the agent cannot get.
