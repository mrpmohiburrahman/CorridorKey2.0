# PR15 on a 16 GB M1 Pro

## Scope and decision

This review covers PR15 at `ccb897c703eb272212323f3d45683d577bfb2e84` against local master `485f5e9ab759ead6138cc990776184b59d6ea971`. [PR15][pr15]

The target is 1920x1080 SDR video on an M1 Pro with 16 GB unified memory. The target codec is ProRes 4444 with straight alpha.

This review used source inspection and primary documents. It did not load a model, install packages, download weights, run inference, or create test media.

This review claims no M1 Pro result. PR15 reports results for a 24 GB M2 with Python 3.12 and PyTorch 2.9.1. [PR15 validation][pr15]

**Decision:** PR15 supplies the correct Python base for Apple Silicon. PR15 alone does not establish a reliable Resolve round trip on this Mac.

The first release needs these additions:

- A preflight that stops on insufficient memory, excess swap, excess CPU, Resolve activity, or insufficient disk space
- Conservative Metal limits for explicit low-memory mode and automatic mode
- Native MPSGraph attention off for the first test
- An explicit sRGB intermediate and Rec.709 final-output color route
- Exact frame rate and source-timecode mapping outside the current writer
- A later approval-gated test on this M1 Pro

PR17 does not replace PR15 for this release. PR17 lacks Apple measurements and adds a large Rust alternative.

## Evidence labels

- **Sourced** means a primary document or source file states the fact
- **Code inference** means source logic supports the conclusion, but no device test proves it
- **Estimate** means arithmetic or a planning limit, not a measurement
- **Unresolved** means the available evidence cannot decide the issue
- **Recommendation** means a proposed release control

## What PR15 changes

### Device execution

**Sourced:** PR15 prefers CUDA, then selects MPS. Master rejects every system without CUDA. [Device selection][pr-device] and `infer.py:120-121`.

**Sourced:** PR15 replaces device-specific CUDA phase moves with the selected device. It retains the execution device while modules reside on the CPU. [Phase movement][pr-runtime-phase] and [device tracking][pr-device-track].

**Sourced:** PR15 uses `non_blocking=True` only for CUDA. Every MPS upload uses a synchronous copy. [RVM upload][pr-rvm-upload], [stage 1 uploads][pr-stage1-upload], and [stage 2 uploads][pr-stage2-upload].

**Code inference:** This change addresses the PR10 corruption mechanism. It does not prove correct M1 Pro output.

### Unsupported area resize

**Sourced:** PR15 moves non-divisible MPS area resize to the CPU. It uses the same PyTorch operator and retains the input dtype. [Area resize][pr-area-resize].

**Code inference:** Native 1080p uses a 1152x1920 model canvas because master pads height to a multiple of 128. `corridorkey/video_io.py:198-207` and `corridorkey/runtime.py:105`.

**Code inference:** The MoGe resize target is 252x420. Since 1152 does not divide by 252, this path uses the CPU fallback. `corridorkey/runtime.py:283-292`.

### Memory controls

**Sourced:** Automatic MPS mode sets a 0.5 high watermark and a 0.4 low watermark when neither variable exists. [Metal limits][pr-metal-limits].

**Sourced:** PyTorch defines the high watermark as a hard allocation limit. The low watermark is a soft target for garbage collection and command-buffer commits. [MPS environment variables][pt-mps-env].

**Sourced:** PR15 computes its MPS budget from 25 percent of recommended memory, driver allocations, and live tensor allocations. [MPS budget][pr-mps-budget].

**Sourced:** PyTorch defines live tensor memory without cached pools. Driver memory includes cached pools and MPSGraph allocations. [Live memory][pt-current-memory] and [driver memory][pt-driver-memory].

**Sourced:** PR15 moves foundations, stage 1, and stage 2 between the CPU and MPS in low-memory mode. It also writes features to disk-backed memmaps. [Low-memory phases][pr-runtime-phase] and [feature memmaps][pr-feature-memmaps].

**Sourced:** PR15 bounds 2D Swin windows, CSwin strips, full-spatial queries, and MLP tokens. [Runtime limits][pr-runtime-limits], [Swin batching][pr-swin-batch], [CSwin batching][pr-cswin-batch], and [query chunking][pr-query-chunk].

**Code inference:** These limits bound attention intermediates. They do not bound every live QKV tensor, output tensor, CPU copy, memmap page, or FFmpeg process.

### Optional MPSGraph attention

**Sourced:** The native operator handles unmasked full-spatial attention. It requires macOS 15 or later and a library built for the exact PyTorch version. [Native loader][pr-native-loader].

**Sourced:** The operator normalizes offset or noncontiguous buffers. It synchronizes Metal before returning output. [Native buffers][pr-native-buffers] and [native synchronization][pr-native-sync].

**Sourced:** PR15 reports different floating-point output between native and PyTorch attention. [PR15 validation][pr15].

**Recommendation:** Keep native attention off for the first test. Do not build the native library for that test.

### Dependencies

**Sourced:** PR15 specifies native arm64 Python 3.12, `torch==2.9.1`, and `torchvision==0.24.1`. [Mac setup][pr-mac-setup]. PyTorch also lists the 2.9.1 installation family in its official version history. [PyTorch versions][pt-versions].

**Sourced:** The Mac requirements select NumPy 2.x and `opencv-python==4.13.0.92`. Other dependencies retain broad ranges. [Mac requirements][pr-requirements].

**Recommendation:** Use a separate arm64 Python 3.12 environment. Do not use the current Python 3.14.7 environment for the first release.

**Unresolved:** Python 3.14.7 compatibility is not established. Compatibility with macOS 26.6.2 and every transitive dependency also remains untested.

**Recommendation:** Lock all direct and transitive versions after package metadata review. Do not install packages during static review.

**Recommendation:** Check for FFmpeg with `prores_ks`, raw BGRA64 input, and YUVA444P10LE output support. FFmpeg documents the `4444` profile and alpha controls. [FFmpeg ProRes][ffmpeg-prores].

## Memory risks and required gates

### PR15 can continue after a failed estimate

**Sourced:** The selector returns Hann 1/1 when no profile fits. The caller prints a warning but continues inference. [Profile fallback][pr-profile-fallback] and [warning path][pr-memory-warning].

**Recommendation:** Change this path to stop before the first model window. A warning is not a sufficient 16 GB release gate.

### Explicit low-memory mode skips automatic Metal limits

**Sourced:** `--low_vram` or an explicit Hann argument disables automatic mode. PR15 calls `configure_auto_memory` only in automatic mode. [Automatic-mode decision][pr-auto-decision].

**Code inference:** Explicit low-memory mode can retain PyTorch defaults of 1.7 high and 1.4 low on unified memory. [MPS environment variables][pt-mps-env].

**Recommendation:** Apply 0.5 high and 0.4 low before every MPS model allocation. Preserve an explicit user setting only after a preflight check.

### The budget does not measure systemwide free memory

**Code inference:** The formula excludes CPU tensors, OpenCV buffers, memmap pages, separate FFmpeg processes, and other applications. PyTorch documents the per-process scope of its MPS APIs. [MPS APIs][pt-mps-apis].

**Recommendation:** Treat `memory_budget_gb` as one gate. Add system memory pressure, swap, CPU, process, and disk gates.

### Hann settings for this Mac

**Estimate:** At 1152x1920, the PR15 CUDA reference table gives these low-memory peaks. The values use its 2.21-megapixel row. [Peak table][pr-peak-table].

| Hann chunk/stride | Reference peak | Peak times 1.1 plus 0.3 GB |
|---|---:|---:|
| 1/1 | 2.80 GB | 3.38 GB |
| 2/1 | 2.97 GB | 3.57 GB |
| 4/2 | 3.35 GB | 3.99 GB |
| 8/4 | 4.12 GB | 4.83 GB |

These are not MPS measurements. They are not M1 Pro guarantees.

**Recommendation:** Use Hann 1/1 for the first three-frame model test. Use low-memory mode. Use stage 2 batch 1.

**Recommendation:** Test 2/1, 4/2, then 8/4 only after separate approval. Stop at the first failed gate or visible continuity regression.

**Recommendation:** Do not start with the 16 GB table entry of 48/24. That table came from CUDA reference measurements, not this M1 Pro.

**Code inference:** Hann 1/1 removes stage 1 temporal window context. The RVM prepass still processes frames in sequence with recurrent state. [RVM state][pr-rvm-state] and [Hann behavior][pr-hann-docs].

### First-test batch settings

**Recommendation:** Use these values with native attention off:

| Setting | First-test value |
|---|---:|
| `hann_chunk` | 1 |
| `hann_stride` | 1 |
| `low_vram` | on |
| `low_vram_foundation_chunk` | 1 |
| `low_vram_enc_frame_chunk` | 1 |
| `low_vram_swin_window_batch` | 1 |
| `low_vram_full_attn_query_chunk_tokens` | 128 |
| `low_vram_mlp_chunk_tokens` | 4096 |
| `stage2_batch` | 1 |
| `model_scale` | 1.0 |
| `ffmpeg_threads` | 2 |

These values match the conservative automatic MPS branch when its budget is below 6 GB. [Small-Mac limits][pr-small-mac-limits].

### Preflight and runtime limits

**Recommendation:** Stop before model loading unless every gate passes:

- DaVinci Resolve has no running process
- Real model and foundation assets exist with approved sizes and hashes
- Sustained total CPU use is at most 80 percent
- Swap use is at most 6 GB
- macOS reports no memory-pressure warning
- The selected Hann estimate fits the PR15 budget with its full margin
- Native attention is off
- The temporary volume has enough free space
- The exact environment and source revision match the approved lock

**Recommendation:** Require at least 3.38 GB of PR15 heuristic budget for Hann 1/1 at this resolution. This is an estimate, not free-RAM evidence.

**Recommendation:** Recheck the budget after model loading and before the first window. Stop if the budget decreases below the required margin.

**Recommendation:** Sample total CPU, swap, memory pressure, process RSS, and MPS driver memory during each phase. Include child processes.

**Recommendation:** Pause for approval when sustained CPU exceeds 80 percent or swap exceeds 6 GB. Do not continue automatically.

**Recommendation:** Stop on nonfinite output, a Metal out-of-memory error, a decode count mismatch, or an encoder error.

### Temporary disk use

**Sourced:** Low-memory mode creates RVM, foundation, and baton memmaps. Their dimensions depend on frame count and model channels. [Memmap shapes][pr-memmap-shapes].

**Estimate:** At 1152x1920, default `d_16=512`, MoGe width 768, RVM width 384, and no C-RADIO give about 10.42 MiB per frame. Local defaults appear in `corridorkey/model.py:46-95`, `corridorkey/stage1.py:238-247`, and `corridorkey/rvm.py:49-58`.

**Estimate:** This example gives about 31.3 MiB for three frames and 3.07 GiB for 302 frames. C-RADIO increases the total. Actual checkpoint metadata can change the total.

**Recommendation:** Calculate disk needs from real checkpoint metadata. Add output files and a 20 percent planning margin. The 20 percent value is an estimate.

## Precision findings

**Sourced:** PyTorch autocast chooses dtypes by operation. It does not cast every operation to bfloat16. [AMP documentation][pt-amp].

**Sourced:** PyTorch documents float intermediates for math SDPA with float16 or bfloat16 inputs. Different backends can produce different output. [SDPA documentation][pt-sdpa].

**Sourced:** PR15 uses bfloat16 autocast for foundations, stage 1, and stage 2. It converts alpha and foreground rendering to float32. [Autocast use][pr-autocast-use] and [render conversion][pr-render-float].

**Unresolved:** The PyTorch 2.9 AMP page has no MPS-specific operation eligibility table. It does not prove every required operation on this M1 Pro.

**Recommendation:** Check tiny MPS bfloat16 operations before model loading in the later test. Include matrix multiplication, convolution, RMS norm, SDPA, resize, and CPU copies.

**Recommendation:** Stop on unsupported or nonfinite bfloat16 operations. Do not silently substitute float16 or float32.

## Resolve export settings

The Resolve 21.1 manual distinguishes single-clip and individual-clip rendering. Individual clips can retain source timecode, source frame rate, and source resolution. [Resolve manual][resolve-manual], pp. 4113-4114.

Use this route for each source shot:

1. Set Color Science to DaVinci YRGB Color Managed.
2. Assign the correct Input Color Space to the source.
3. Set the timeline to Rec.709 Gamma 2.4.
4. Set Output Color Space to sRGB for the processing master only.
5. Open Deliver with a custom setup.
6. Select Individual Clips.
7. Enable Export Video.
8. Select QuickTime and Apple ProRes 4444.
9. Enable Render at Source Resolution.
10. Check for 1920x1080 output.
11. Retain the source frame rate.
12. Use progressive frames and square pixels.
13. Disable timeline effects unless the shot requires them.
14. Disable data burn-in.
15. Disable optimized media and proxy media.
16. Set Data Levels to Auto.
17. Set Color Space Tag to Rec.709.
18. Set Gamma Tag to sRGB or IEC 61966-2-1.
19. Use source filenames with separate folders or unique names.
20. Retain handles and record each source in-point.

The manual states that source resolution and source frame rate support round-trip workflows. [Resolve manual][resolve-manual], p. 4114.

The manual states that output color space controls the rendered transform. Color and gamma tags write metadata. [Resolve manual][resolve-manual], pp. 226-229 and 4119.

**Recommendation:** Keep Bypass Re-encode off for the sRGB processing master. A color transform requires new pixels.

**Recommendation:** Use ProRes 422 HQ only when 4:4:4 source chroma is unnecessary. ProRes 4444 preserves full-resolution 4:4:4:4 color and alpha. [Apple ProRes][apple-prores], pp. 3-4.

**Unresolved:** Auto data levels can still conceal a source metadata error. Check a gray ramp and color patches in the later test.

**Recommendation:** Do not select Full on only one side of the exchange. Resolve requires the receiving application to use the same range. [Resolve manual][resolve-manual], pp. 221-224 and 4118-4119.

## CorridorKey process settings

Use the PR15 Python path with native attention off. Use `--outputs cutout` and `--model_scale 1.0`.

Leave `--cutout_linear` off. The default video path writes display-coded sRGB RGB with separate alpha. `corridorkey/runtime.py:686-730`.

Use `--despill_strength 0` for the first color test. This avoids an intentional RGB change during the color check.

Choose the production despill value explicitly. The default 0.5 changes foreground RGB. `corridorkey/runtime.py:604-614` and `infer.py:89-90`.

The existing writer concatenates straight RGB with alpha. It does not multiply RGB by alpha. `corridorkey/runtime.py:623-628` and `corridorkey/runtime.py:749-756`.

The writer uses `prores_ks`, profile 4, and `yuva444p10le`. It tags Rec.709 primaries and matrix with sRGB transfer. `corridorkey/video_io.py:624-660`.

**Code inference:** The current command does not explicitly control the RGB-to-YUV conversion matrix or range. Metadata flags alone do not establish pixel conversion.

**Recommendation:** Make the BT.709 matrix and range conversion explicit in the writer. Keep alpha separate during that conversion.

**Recommendation:** Pin the FFmpeg build and record the `prores_ks` alpha-bit setting. Check actual encoded alpha depth in the later media test.

Apple states that ProRes 4444 color is visually lossless. Alpha compression is mathematically lossless up to 16 bits. [Apple ProRes][apple-prores], p. 18.

This does not mean model alpha remains unquantized. The current writer supplies 10-bit YUVA pixels to FFmpeg.

## Resolve import settings

1. Import the cutout into the color-managed Rec.709 project.
2. Open Clip Attributes for the cutout.
3. Set Video Frame Rate to the exact source rate.
4. Set Data Levels to Auto.
5. Set Alpha Mode to Straight.
6. Do not invert alpha.
7. Set Input Color Space to sRGB.
8. Retain Rec.709 Gamma 2.4 as the final project output.
9. Align the cutout with the recorded source in-point.
10. Keep original audio on the source timeline.

The manual lists ProRes 4444 as an alpha-capable format. Clip Attributes supports None, Straight, Premultiplied, and inverted alpha. [Resolve manual][resolve-manual], pp. 448-451.

The extracted local manual pages contain the same controls on printed pages 450-451. `/Applications/DaVinci Resolve/DaVinci Resolve Manual.pdf`.

Apple defines straight alpha as separate from RGB. Premultiplied alpha also multiplies the visible RGB channels. [Apple alpha guide][apple-alpha].

**Recommendation:** Do not select Premultiplied for this writer. That setting can cause incorrect edges.

**Recommendation:** Do not select None. That setting disables transparency.

The current writer disables audio with `-an`. It supplies no source-timecode option. `corridorkey/video_io.py:630-660`.

**Code inference:** Resolve cannot recover original audio or source timecode from the cutout alone. Use the original timeline audio and a source-mapping manifest.

**Recommendation:** Record source filename, source in-point, source out-point, exact frame rate, frame count, color route, and data range.

**Unresolved:** OpenCV supplies floating-point FPS. Long-clip timing drift remains untested. Use an exact rational frame rate in a later adaptation.

## Smallest suitable adaptation

**Recommendation:** Port the PR15 Python device, runtime, and 2D batching changes as one unit. Keep native MPSGraph disabled by default for this release.

**Recommendation:** Add a named 16 GB MPS profile with the first-test values above. Apply Metal watermarks for both automatic and explicit modes.

**Recommendation:** Make an insufficient memory estimate fatal. Add the system and disk gates before model loading.

**Recommendation:** Keep the existing straight-alpha ProRes writer. Add explicit color conversion, exact FPS mapping, and a source manifest.

**Recommendation:** Use the managed sRGB intermediate route. Do not relabel sRGB pixels as Rec.709 Gamma 2.4 without a transform.

**Recommendation:** Do not add PR17 or a native build to the first release.

## Missing tests and correctness risks

PR15 adds focused device, window-batching, and native-operator tests. [Device tests][pr-tests-device], [window tests][pr-tests-window], and [native tests][pr-tests-native].

The following evidence is missing:

- An actual 1080p run on a 16 GB M1 Pro
- A test with the real weights and foundation assets
- A test of the production 1152x1920 CPU resize fallback
- A fail-closed test for insufficient MPS budget
- A Metal-limit test for explicit low-memory mode
- A systemwide memory test including FFmpeg and memmap pages
- A ProRes 4444 Resolve round trip with semi-transparent edges
- A BT.709 matrix, transfer, range, and gray-ramp test
- An exact frame-rate and source-timecode test
- A long-clip continuity test for Hann 1/1
- A Python 3.14 compatibility test
- An M1 Pro native-MPSGraph comparison

The PR15 memory samples exclude separate FFmpeg processes and can miss short spikes. [PR15 validation][pr15].

## Safe static checks for later approval

Do not run these checks during this research task.

1. Check the PR head and base commit identifiers.
2. Run `git diff --check` on the adapted source.
3. Parse every changed Python file with `ast.parse`.
4. Compare CLI argument definitions through AST inspection.
5. Check that remaining CUDA calls are inside CUDA-only branches.
6. Check that every MPS upload uses synchronous copies.
7. Check that native attention defaults off in the release profile.
8. Inspect FFmpeg arguments with a mocked `subprocess.Popen`.
9. Assert ProRes profile 4, YUVA444P10LE, two threads, and no audio.
10. Assert explicit BT.709 conversion and range arguments after adaptation.
11. Run CPU-only window-parity tests with small tensors.
12. Test 1080p profile selection with mocked MPS memory APIs.
13. Test that insufficient budget raises an error before model loading.
14. Inspect package metadata without installing packages.
15. Check asset paths and hashes without loading tensors.
16. Check temporary-disk arithmetic with real checkpoint metadata.
17. Check that no generated native library exists for the first test.

## Approval-gated heavy test plan

Do not run this plan during static review.

### Stage 1: tiny media writer test

1. Close Resolve and other GPU applications.
2. Approve one three-frame 1080p writer test.
3. Use RGB patches, a gray ramp, and alpha values of 0, 0.5, and 1.
4. Encode through the adapted writer without model loading.
5. Decode every frame and check dimensions, count, FPS, alpha, matrix, transfer, and range.
6. Import the file into Resolve with Straight alpha.
7. Compare composites over black, white, and a checkerboard.
8. Stop on color shifts, halos, missing alpha, or timing errors.

### Stage 2: tiny MPS operator test

1. Approve small MPS bfloat16 operator checks.
2. Check each required operation for finite output.
3. Compare CPU area resize with the MPS fallback.
4. Check synchronous CPU-to-MPS copies for stable output.
5. Stop on unsupported operations or corruption.

### Stage 3: three-frame model test

1. Install only the approved locked environment.
2. Supply only approved real assets.
3. Close Resolve.
4. Set `CORRIDORKEY_MPS_NATIVE_ATTENTION=0`.
5. Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5`.
6. Set `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4`.
7. Run three frames with Hann 1/1 and the conservative batch table.
8. Use cutout output, stage 2 batch 1, native resolution, and zero despill.
9. Monitor every preflight and runtime gate.
10. Check all output frames for finite alpha and foreground values.
11. Repeat the Resolve import and composite checks.
12. Record peak RSS, MPS driver memory, swap, CPU, elapsed time, and disk use.

A later command can use this shape after approval:

```sh
CORRIDORKEY_MPS_NATIVE_ATTENTION=0 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 \
python3.12 infer.py \
  --input /path/to/processing_master.mov \
  --output_dir /path/to/output \
  --outputs cutout \
  --num_frames 3 \
  --start_mode begin \
  --hann_chunk 1 \
  --hann_stride 1 \
  --low_vram \
  --low_vram_foundation_chunk 1 \
  --low_vram_enc_frame_chunk 1 \
  --low_vram_swin_window_batch 1 \
  --low_vram_full_attn_query_chunk_tokens 128 \
  --low_vram_mlp_chunk_tokens 4096 \
  --stage2_batch 1 \
  --model_scale 1.0 \
  --despill_strength 0 \
  --ffmpeg_threads 2
```

### Stage 4: bounded expansion

1. Request approval before each Hann increase.
2. Test 2/1, then 4/2, then 8/4.
3. Retain the largest setting that passes memory and continuity checks.
4. Request separate approval before any native MPSGraph build.
5. Compare native-off and native-on output only after the native-off baseline passes.
6. Test a longer clip only after all short tests pass.

## Release limits until tests pass

- No claim of M1 Pro support
- No claim of exact CUDA and MPS pixel parity
- No claim of source-timecode or audio preservation
- No claim of bit-exact model alpha
- No native MPSGraph requirement
- No automatic continuation after memory or swap limits
- No full-length media test without separate approval

## Sources

[pr15]: https://github.com/nikopueringer/CorridorKey2.0/pull/15
[pr-device]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/device.py#L10-L15
[pr-device-track]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/device.py#L18-L22
[pr-runtime-phase]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L436-L454
[pr-rvm-upload]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L360-L371
[pr-stage1-upload]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L542-L565
[pr-stage2-upload]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L915-L930
[pr-area-resize]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/device.py#L37-L42
[pr-metal-limits]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/device.py#L25-L34
[pr-mps-budget]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/device.py#L61-L71
[pr-feature-memmaps]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L380-L433
[pr-runtime-limits]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L295-L310
[pr-swin-batch]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/blocks.py#L1173-L1190
[pr-cswin-batch]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/blocks.py#L2034-L2043
[pr-query-chunk]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/blocks.py#L2486-L2531
[pr-native-loader]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/mps_attention.py#L11-L33
[pr-native-buffers]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/mps_attention.py#L36-L58
[pr-native-sync]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/native/mps_attention.mm#L93-L102
[pr-mac-setup]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/README.md#L19-L57
[pr-requirements]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/requirements.txt#L1-L14
[pr-profile-fallback]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L174-L189
[pr-memory-warning]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L221-L227
[pr-auto-decision]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/infer.py#L134-L139
[pr-peak-table]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L134-L150
[pr-rvm-state]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L332-L376
[pr-hann-docs]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/README.md#L265-L298
[pr-small-mac-limits]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L192-L220
[pr-memmap-shapes]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L332-L345
[pr-autocast-use]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L360-L371
[pr-render-float]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/corridorkey/runtime.py#L679-L720
[pr-tests-device]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/tests/test_device.py#L1-L132
[pr-tests-window]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/tests/test_window_batching.py#L1-L78
[pr-tests-native]: https://github.com/nikopueringer/CorridorKey2.0/blob/ccb897c703eb272212323f3d45683d577bfb2e84/tests/test_mps_attention.py#L1-L46
[pt-mps-env]: https://docs.pytorch.org/docs/2.9/mps_environment_variables.html
[pt-current-memory]: https://docs.pytorch.org/docs/2.9/generated/torch.mps.current_allocated_memory.html
[pt-driver-memory]: https://docs.pytorch.org/docs/2.9/generated/torch.mps.driver_allocated_memory.html
[pt-mps-apis]: https://docs.pytorch.org/docs/2.9/mps.html
[pt-versions]: https://pytorch.org/get-started/previous-versions/
[pt-amp]: https://docs.pytorch.org/docs/2.9/amp.html
[pt-sdpa]: https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html
[ffmpeg-prores]: https://ffmpeg.org/ffmpeg-codecs.html#ProRes
[apple-prores]: https://www.apple.com/final-cut-pro/docs/Apple_ProRes.pdf
[apple-alpha]: https://support.apple.com/guide/motion/alpha-channels-motn125277a3/mac
[resolve-manual]: https://www.blackmagicdesign.com/support/family/davinci-resolve-and-fusion
