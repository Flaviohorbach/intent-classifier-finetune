# AI log: what the agent got wrong

Built with a coding agent (Claude Code), as in [project 5](https://github.com/Flaviohorbach/legacy-storefront-refactor). These happened while building this project.

### 1. The evaluation measured a slightly different model than the page runs

Training embedded messages in batches of 32; the page embeds one at a time. The 8-bit model scales activations per batch, so the same text produced a slightly different vector (cosine 0.995 instead of 1). "Bom dia" came out as a question in the evaluation and as "other" on the page. **Caught by:** typing 20 test messages into the page and comparing with the saved predictions, then embedding one message alone and inside a batch. **Fix:** training embeds one text at a time. The score went from 83.3% to 82.5%, and the page and evaluation now agree on 118 of 120.

### 2. A notebook written for a machine it could not test on

The notebook failed seven times across the user's runs before it finished: a model-loading error whose message was never seen (fixed as a renamed argument, `torch_dtype` → `dtype`, without proof), Colab's old `peft` refusing its newer `transformers`, the new `peft` refusing an old `torchao`, `warmup_ratio` removed in transformers 5, a batch of 16 that ran out of the T4's 15 GB, and, on Kaggle, two cells that assumed `import google.colab` fails outside Colab (Kaggle ships that package). Each fix came from the error the user pasted. **Caught by:** the user running it. **Fix:** version-independent code, batches of 4 with gradient accumulation, and checking an environment variable instead of an import. A notebook pinned to tested versions, and a dry run on a tiny model on CPU, would have caught most of these before the user spent GPU time on them.

### 2b. Memory that stayed taken after a failure

After the first out-of-memory error, a smaller batch failed again with 13.8 GB already in use. The failed run's traceback kept its GPU tensors alive, so freeing memory inside the session did not help. **Caught by:** the second error asking for only 198 MB. **Fix:** restarting the session. The agent should have said so on the first error instead of offering an in-place retry.

### 3. An upgrade that may have broken the GPU

One fix added `pip install -U peft accelerate`, and the next run said PyTorch had no CUDA. The upgrade was the suspect, so it was removed, but the runtime may simply have reconnected without a GPU: the cause was not proven. **Caught by:** the user's next run. **Fix:** upgrade only what is needed.

### 4. A wrong number in a progress message

Reporting the keyword baseline, the agent compared 52.5% here with "77% on project 9". That 77% was project 9's routing score; on the same task, picking the label, keywords got 20 of 30 there (67%). **Caught by:** rereading project 9's results while writing the README. **Fix:** the README uses 20 of 30.

### 5. Label noise noticed by reading, not by a metric

The generated data looked fine in counts and balance. Reading a random sample showed a message about a cut hand labeled as an exchange, and counting found that serious complaints were mostly about money (72) and rarely about health (17). **Caught by:** reading samples before training. **Fix:** none in the data, on purpose: correcting it after seeing the test results would bias them. The README explains the gap instead.

## What needed a human

- Choosing both a browser model and a LoRA model, and synthetic data with a hand-written test.
- Running the notebook on Colab, which needs a Google account and a free GPU the agent cannot get.
