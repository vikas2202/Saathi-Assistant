# Build verification — 2026-09-27

## Completed

- Created an isolated Python 3.14 environment in `.venv` and installed the default dependencies.
- Downloaded YuNet and SFace weights and verified their SHA-256 hashes.
- Passed 37 automated tests using `unittest`.
- Passed hidden real-Tk UI checks: greeting, session isolation with multiple faces, stale reply rejection, enrollment cancellation and camera-stop cleanup.
- Ran real YuNet inference on a synthetic blank frame and SFace feature inference producing 128 finite values.
- Passed dependency consistency checks (`pip check`) and application diagnostics (`doctor`).
- Opened the desktop app with the camera and microphone off.

The desktop screenshot API failed twice with `SetIsBorderRequired failed: No such interface supported (0x80004002)`. Visual inspection was not completed; programmatic UI checks passed.

## Still requires user data or hardware interaction

- Live camera recognition with enrolled people and unknown volunteers.
- Real microphone transcription and speaker playback.
- Live cloud conversation using the user's API key/account.
- Representative tuning and held-out evaluation captures before choosing personalized thresholds.
- Optional YOLO/PyTorch module installation and performance testing if enabled.

No participant has been enrolled, no face accuracy claim is made, and no network weights have been fine-tuned. Automated cloud tests use a mocked HTTP transport and incur no API usage. Device hardware behavior, liveness and multi-speaker attribution are not established by these tests.
