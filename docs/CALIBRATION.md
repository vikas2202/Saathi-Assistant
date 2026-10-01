# Camera-specific calibration and future fine-tuning

## What is implemented

The shipped SFace network stays fixed. The implemented adaptation searches the **recognition similarity threshold** and **identity margin** using labeled camera embeddings. This is threshold calibration, not neural-network training.

The matcher uses the maximum cosine similarity over each profile's stored templates. It rejects a result below the threshold or too close to another person's score. Calibration evaluates exactly this decision rule. The UI additionally requires stable frames and good image quality, so live session behavior still needs separate testing.

## Collect usable data

1. Enroll willing participants with distinct names. Do not change those profiles during the evaluation.
2. Capture **tune** sessions for each enrolled person and several consenting people who are not enrolled.
3. Capture **test** sessions separately, ideally later with realistic lighting, distance and appearance changes. Do not re-use enrollment frames or copy tuning frames into test data.
4. Keep only the labeled participant in view. The capture tool cannot independently prove your label is correct.
5. Use several sessions and several participants per group. The minimum of ten rows per split is only a software check, not sufficient evidence of deployment accuracy. Adjacent video frames are correlated.

Close the desktop app's camera first. From the project directory:

```powershell
.venv\Scripts\python.exe -m saathi capture-eval --name Saiyam --split tune --count 20 --consent
.venv\Scripts\python.exe -m saathi capture-eval --unknown --split tune --count 20 --consent
# Later, in genuinely separate recording sessions:
.venv\Scripts\python.exe -m saathi capture-eval --name Saiyam --split test --count 20 --consent
.venv\Scripts\python.exe -m saathi capture-eval --unknown --split test --count 20 --consent
```

Repeat for additional people and conditions. `--unknown` means the participant has no enrolled profile. `--consent` records your confirmation that the participant agreed; obtain their permission before running it. Captures save embeddings, labels and session identifiers to `data/evaluation.jsonl`, with no photos/audio. Cancelling or timing out discards the incomplete session.

## Measure, then apply

```powershell
.venv\Scripts\python.exe -m saathi calibrate
# Review calibration-report.json, then apply if it is acceptable:
.venv\Scripts\python.exe -m saathi calibrate --apply
```

The search selects settings using only the tune split, targeting observed unknown false accepts at or below 1%, zero observed wrong-identity assignments, and at least 80% correct known-person recognition. These are prototype acceptance targets, not validated service guarantees. `--max-far` changes the observed false-accept target.

The held-out test split is evaluated after selection. Applying is refused if those false-match or known-recognition targets fail. Repeatedly changing settings after looking at the same test results overfits that test; collect a new independent test set after changes.

Reports include:

- **False accept:** an unknown participant matched an enrolled profile.
- **False reject:** an enrolled participant was labeled unknown.
- **Wrong identity:** an enrolled participant matched another enrolled profile.
- **Known recognition rate:** fraction of enrolled-person samples matched correctly.
- Counts, sample denominators, tune/test separation, and a profile-template fingerprint.

Exact duplicate embeddings, shared tune/test session IDs, and exact enrollment-template reuse are rejected. These checks cannot detect every near-duplicate or mislabeled capture. Manually verify session separation and labels.

## Improving a specific failing case

| Observed failure | First intervention | Verify using |
|---|---|---|
| Saiyam unknown in side lighting | Improve camera placement and collect varied enrollment views | New side-lighting tune and test sessions |
| Two people confused | Increase threshold/margin; improve templates for both people | Both enrolled people and similar-looking unknown volunteers |
| Glasses or hairstyle changes | Re-enroll with representative current views | Separate sessions with and without the change |
| Low FPS | Disable YOLO, lower resolution, reduce processing rate | Measure end-to-end FPS and recognition errors together |
| Name transcription errors | Review/correct transcript; improve microphone distance | Name utterances in the intended language and acoustic setting |
| Overly long or irrelevant replies | Adjust the conversation instructions and test cases | A fixed, human-reviewed conversation evaluation set |

## Actual model fine-tuning

Do not fine-tune on a few webcam frames from one person. First establish that calibration, image quality and representative enrollment cannot solve the measured problem.

Face-encoder training needs a larger consented, identity-labeled dataset, an appropriate training implementation, identity-disjoint evaluation for generalization, compute, and bias/error analysis. An updated encoder changes the embedding space: version the model and re-enroll/migrate templates rather than comparing incompatible embeddings. This repository does not include a face-encoder training job.

Language-model fine-tuning needs representative, reviewed conversation examples with target responses, scenario-separated training/validation/evaluation sets, and account support. Do not use fine-tuning as personal memory: keep names and preferences in the editable database. Do not upload biometric data or private histories as conversation-training examples.

No fine-tuning job, paid API call, or user dataset upload was performed while building this project. Add your key in the application to test live cloud conversations.
