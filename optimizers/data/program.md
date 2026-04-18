# Autoresearch: Traditional Image Processing

This is an experiment to have the LLM act as an autonomous researcher to optimize traditional image processing algorithms (e.g., image denoising, enhancement) running on CPU.

---
## Experiment Execution Mode: AUTO (No User Confirmation Required)

When starting the experiment, you MUST:
1. Execute all steps directly without asking any confirmation questions
2. Do NOT wait for user replies such as "yes/proceed/continue"
3. Automatically choose reasonable default values (e.g., use current date format YYYY-MM-DD for tag names)
4. Attempt to auto-fix issues when encountered; only report errors when they cannot be automatically resolved

# Description
Key Task Description:{{SCENARIO_DESCRIPTION}}

# The train file path
Train File Path:{{TRAIN_FILE_PATH}}

# The user image file path
User Image File Path:{{USER_IMAGE_FILE_PATH}}

# The prepare file path
Prepare File Path:{{PREPARE_FILE_PATH}}

## Setup

1. **Read the in-scope files**:
   - `prepare.py` — fixed constants, test data generation, and the evaluation metric (e.g., MSE calculation). **Do not modify.**
   - Target File (`{{TRAIN_FILE_PATH}}` or as specified above) — the file you modify. It contains the image processing pipeline and hyperparameters.
2. **Create a backup of the target file**: Before any modification, save the original content of the target file so it can be restored later. Use a file-based backup mechanism (e.g., copy the target file to a `.bak` path).

Once you are ready, kick off the experimentation immediately.

## Experimentation
The training script runs for a **fixed time budget of 2 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python "{{TRAIN_FILE_PATH}}" --image-path "{{USER_IMAGE_FILE_PATH}}" --output-path output.png --scene-prompt "{{SCENARIO_DESCRIPTION}}"`.

**What you CAN do:**
- Modify the target file (`{{TRAIN_FILE_PATH}}` or as specified above).
- Everything in the target file is fair game: change hyperparameters (kernel size, sigma, etc.), swap out algorithms (e.g., Gaussian vs. Median vs. Bilateral), or combine multiple OpenCV/NumPy filters into a stronger pipeline.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only and contains the ground truth evaluation metric.
- Run any `git` commands. This environment is not managed by git.
- Install new packages or add dependencies unless explicitly permitted. Use what's already available (e.g., `opencv-python`, `numpy`, `Pillow`).

**The Goal:**
Achieve the best possible evaluation score (e.g., lowest MSE).
- **Performance:** CPU execution should be fast. If a complex filter takes too long (e.g., > 30 seconds per image), it might not be worth it.
- **Simplicity criterion:** All else being equal, simpler is better. A 0.1% improvement that adds 50 lines of complex masking logic is not worth it.

**The first run**: Your very first run should always be to establish the baseline. Run the target script as is and record the result.

## Logging results

When an experiment is done, output a JSON summary to stdout so the caller can capture the result. Do NOT write `results.tsv` to disk.

The JSON object must have these exact fields:

```json
{
  "round": 1,
  "metric_score": 123.45,
  "execution_ms": 456.78,
  "status": "keep",
  "description": "switched to median blur k=5"
}
```

Field meanings:
- `round`: round number (1 or 2)
- `metric_score`: The evaluation metric achieved (e.g., MSE). Use `999999` for crashes.
- `execution_ms`: Algorithm execution time in milliseconds.
- `status`: `"keep"`, `"discard"`, `"crash"`, or `"timeout"`
- `description`: short text description of what this experiment tried

## The experiment loop

You have a maximum limit of **2 rounds** of experimentation.

LOOP (Up to 2 rounds):

1. Before modifying code, create a backup of the current target file (e.g., copy to `"{{TRAIN_FILE_PATH}}.bak"`).
2. Tune the target script with an experimental idea by directly hacking the code.
3. Run the experiment with a strict **2-minute timeout**: `python "{{TRAIN_FILE_PATH}}" --image-path "{{USER_IMAGE_FILE_PATH}}" --output-path output.png --scene-prompt "{{SCENARIO_DESCRIPTION}}" > run.log 2>&1`
4. If the execution times out (> 2 minutes), treat it as a failure: restore the target file from the backup copy, output a JSON summary with `status: "timeout"`, and move to the next round.
5. If it finishes within 2 minutes, evaluate the results. **Note: The evaluation function is located in `prepare.py`. You must autonomously decide how to import and call this function to calculate the evaluation metric (e.g., MSE) for your current run. (Its absolute path is {{PREPARE_FILE_PATH}}, you may need to append it to sys.path). Do NOT remove the JSON payload output in the target file.**
6. If the script crashed, read the stack trace. If it's a simple typo, fix it and retry within the same round (still counting as one round). If the idea is fundamentally broken, restore the target file from backup, output a JSON summary with `status: "crash"`, and move to the next round.
7. Output a JSON summary with the results.
8. If the metric improved (e.g., lower MSE), keep the modified target file as the new baseline for the next round.
9. If the metric is equal or worse, restore the target file from the backup copy to the original baseline.
10. After each round, remove the backup file to avoid leaving stale artifacts.

**STOPPING CRITERIA**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Proceed automatically until you reach the **2-round limit**. Once the 2 rounds are completed, output a final JSON summary of the best results and terminate the experiment.