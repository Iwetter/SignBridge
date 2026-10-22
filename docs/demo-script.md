# CPU acceptance and Friday rehearsal

## Before the demo

Launch using the single command in README. Use CPU. Verify camera pixels move, a real hand produces a skeleton, handedness matches the signer, and no-hand input commits nothing. If camera access fails, correct Windows permissions and click Live camera. Do not mistake the synthetic replay for recognition evidence.

Train I/A/N first using two genuinely different sessions. Check a third, untouched session with Evaluate. Extend static coverage to 24 letters and calibrate temporal J/Z. Record MY/NAME/YOUR/WHAT clips with the actual demo signer and framing. Additional signs are optional until these pass.

## Live checks (record actual outcomes)

| Check | Required outcome | Current evidence |
|---|---|---|
| Camera and skeleton | Webcam pixels and stable landmarks on real hand | Awaiting user's local camera test |
| A/I/N | Correct candidate and stable commit | Awaiting personal data/model |
| IAN | Exactly IAN, one character per deliberate pose | Synthetic temporal test passes; live pending |
| HELLO | Exactly HELLO; two deliberate L presentations | Synthetic temporal test passes; live pending |
| Hold/no hand | No repeated held letters, no rest tokens | Synthetic tests; live pending |
| J/Z | Full movement required; static I/D must not substitute | Temporal implementation; real clips pending |
| Four core signs | MY NAME YOUR WHAT correctly committed | Calibrated clips pending |
| Conversation | MY NAME I A N YOUR NAME WHAT → expected text | Synthetic integration passes; live pending |
| Speech | Astra's response audible; camera keeps updating | Worker implemented; audible acceptance pending |
| CPU fallback | No board/failed backend still allows CPU | Automated verification |

## Recruiter demonstration

Say: “SignBridge is a local ASL and fingerspelling tutor with a small calibrated sign vocabulary.”

Select Conversation, clear, and sign MY NAME, then I–A–N, then YOUR NAME WHAT. Pause at the endpoint of each sign. Show the committed sequence, inferred name, and Astra's text/spoken response. No keyboard is needed during this sequence. Use Free spelling to demonstrate HELLO with a deliberate hand release between Ls. Use Alphabet tutor to show feedback.

Only show a benchmark when an actual trained model is running. CPU inference measures the static classifier; camera throughput includes landmark extraction. FPGA round trip is unavailable until a physical endpoint is implemented and measured. Do not display synthetic timings as hardware results.

## Freeze and recovery

Before Friday's rehearsal, retain the tested model/metrics and dataset snapshot. Rehearse once after restarting the application. Keep camera USB, lighting and framing stable. If a model change makes recognition worse, reload the last verified local artifact. If hardware fails, select CPU. If the live camera/recognition cannot be validated, describe the demo limitation explicitly; rehearsal remains a software integration demonstration only.
