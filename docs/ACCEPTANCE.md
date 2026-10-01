# Real-world acceptance checklist

Automated tests exercise code behavior and model loading. They do not replace live camera, speech and recognition evaluation.

| Scenario | Expected behavior |
|---|---|
| Start without a camera | Actionable error; app remains open |
| Missing or corrupt model | Download/checksum guidance; no recognition attempt |
| No face | No named greeting or personalized conversation |
| Unknown face | Introduction prompt after stable visibility; no name guessed |
| Denied enrollment permission | No profile saved |
| Duplicate name / already registered face | Enrollment blocked with guidance |
| Two people during enrollment | Enrollment cancelled; no partial profile |
| Person changes during capture | Enrollment cancelled on embedding inconsistency |
| Too dark / blurry / far / clipped | Quality guidance; no poor sample added |
| Enrolled person returns | Name appears after consecutive matches; spoken greeting once per cooldown |
| Lookalike or low-margin match | Unknown/uncertain; no personalized greeting |
| Two faces during chat | Clear active conversation; cancel voice and ignore pending answer |
| Camera disconnect or stale frame | Clear active identity and pause conversation |
| Person leaves while cloud responds | Ignore the obsolete answer; no playback to a new visitor |
| Microphone denied or disconnected | Error, with typed conversation still available |
| Silence | No audio upload; ask for clearer speech |
| Misheard name | Editable transcript and explicit name confirmation |
| Stop voice | Stop local speech/capture and ignore pending result |
| Invalid key / quota / offline / unavailable model | Sanitized actionable message, with face recognition still working |
| Memory request | Show exact proposed memory; save only after approval for the same active session |
| Delete profile | Delete database face templates and notes; recognition becomes unknown |
| App restart | Saved profiles survive; chat history and manually entered API key do not |
| Photo/video presented to camera | May be accepted: liveness is not implemented; do not use as authentication |
| Off-camera person speaks | May be transcribed: active-speaker detection is not implemented |

Record camera model, resolution, CPU, lighting, participant/session counts, inference latency and observed errors. Evaluate enrolled and unknown participants in separate sessions. Report failures, including false matches, rather than presenting only successful demonstrations.

Cloud testing requires the user's API key. Keep it out of screenshots, test fixtures, terminal output and source control. Use the masked Settings field.
