# Validation record

## Windows setup — 2026-09-11

- Installed the pinned dependencies and pytest in a repository-local Python 3.12.1 virtual environment; `pip check` passed.
- Full suite: 290 passed, six Linux-only tests skipped (fcntl, permission helper, evdev).
- Increased recorder detail-label space after its clipping regression test failed with Windows font metrics.
- Downloaded the hand landmark model; model initialization and blank-frame inference passed.
- Camera 0 processed 66 frames in the four-second inference smoke test; camera thread and model cleanup passed.
- Native Qt window launched, was visually inspected, and closed cleanly. Pynput pointer backend and global hotkey listener initialized successfully; control remained paused.
- Actual injected mouse movement, physical F8/F12 presses, interactive gesture feel, custom command bindings, and mixed-DPI mapping remain unverified.
- Follow-up: fixed model initialization under `pythonw.exe`, where `sys.stderr` is `None`. The windowed launcher produced a real 1280 x 720 preview packet with a healthy hotkey listener and control paused. Regression suite: 292 passed, six Linux-only tests skipped.
- Camera startup follow-up: the default MSMF backend took 32.08 seconds to configure and deliver its first 720p frame, versus 5.34 seconds with DirectShow. Windows now selects DirectShow. The visible GUI, launched through pythonw with preview started automatically, rendered its first 1280 x 720 result in 6.16 seconds with control paused. Startup status now names the pending step and elapsed time. Suite: 293 passed, six Linux-only tests skipped. First-frame FPS is not a steady-state performance measurement.

## Original Linux validation

Tested on the current Linux Wayland host on 2026-09-06.

- Installed editable package with all declared dependencies in the repository's isolated Python 3.12.14 environment; dependency integrity check passed.
- 18 automated tests passed: gesture confirmation, hysteresis, click/drag release, right-click latching, cooldown, scroll dwell, gesture toggles, safe start, calibration validation, emergency release, tracking loss, stale frames, screen clamping, jitter suppression, and initial movement speed limiting.
- MediaPipe initialized its downloaded model and correctly returned no hand for a blank frame.
- The real model detected 21 landmarks and a right-hand classification in Google's public `right_hands.jpg` test image. The image was used only for a local inference/overlay check and is not included in this repository.
- Camera 0 processed 89 frames over four seconds in the direct capture/inference smoke test; capture thread exited and camera/model resources were released.
- Native Wayland Qt GUI launched with live webcam processing and shut down cleanly. Last observed GUI processing rate was 16 FPS; this is a brief sample, not a latency benchmark.
- Visually inspected the desktop layout and hand-skeleton diagnostic overlay using an offscreen render.
- Confirmed this host reports preview-only mode and refuses to enable control when no keyboard is readable for global emergency stop.

Not verified: actual OS mouse injection or global F8/F12 on this host (device permissions unavailable), interactive human gesture accuracy, click/drag feel, different Wayland compositors, Windows/macOS, mixed-DPI desktops, or extended-session performance. Mouse actions were verified with a recording test backend. No mouse control was enabled during validation.

Upper-body tracking remains an optional future extension.

## Precision update

Added regression checks for anchored pinch/drag behavior, retained subpixel motion, invalid detection release, isolated tracking jumps, cursor position on reacquisition, stationary context-menu targeting, and scroll-pose flicker. The earlier keyboard discovery issue was fixed and the readable Logitech keyboard listener and virtual-pointer initialization subsequently passed on this host. Full human gesture accuracy and compositor mouse behavior still require interactive validation; automated input tests use a recording backend.

## Monitor gap handling

Automated checks cover projection onto visible monitor edges, negative coordinates, vertical gaps, offset and mirrored layouts, and pointing/dragging across a large gap without getting stuck. All 35 tests passed after the resolution selector and monitor gap updates. These checks use a recording input backend; physical multi-monitor behavior and Wayland compositor output assignment remain unverified.

## Velocity-dependent gain and drag persistence

Added `tests/test_gain.py` (16 checks) and `tests/test_drag_tracking.py` (29 checks); 255 automated tests pass. Covered: the gain curve's bounds, monotonicity and continuity (swept at 1e-5 resolution, largest step 2e-5), the nominal-speed plateau landing where the pre-gain pointer landed, low-speed travel measured against a no-gain reference path, landmark noise moving the gain by under 0.05, the offset staying inside its cap under 600 frames of mixed motion, and the screen edge staying reachable after slow work. On the drag side: entry still requiring a firm pinch, the wider exit threshold, several degraded frames in a row not releasing, clean tracking resuming without a state reset, immediate release on an opened pinch, latch-then-timeout at the grace period, an emergency stop taking no grace period, zero standing lag on a steady 1800 px/s drag, proportional clamping of a 900 px excursion, confidence-weighted correction, damped blind extrapolation, and convergence rather than snapping on reacquisition. Safety: tracking loss never throwing however fast the drag was, a 450 px reacquisition and a 600 px isolated outlier each moving the cursor under 150 px and producing no throw, predicted positions excluded from the release-velocity window, and one wild sample not setting a throw's direction.

Updated four existing expectations that changed with the new semantics rather than through a defect: a single dropped or nonfinite frame mid-drag now latches instead of releasing, and the reacquisition check in the monitor-gap test now uses a gap longer than the grace period so it still exercises reacquisition.

Not verified: interactive feel. Whether the gain curve's anchors suit a real hand, whether 150 ms is the right grace period, and whether the drag now feels attached all need a human at a webcam. The numbers above were measured from simulated landmark streams through a recording input backend; no desktop pointer was moved.

## Pointer regression: velocity gain replaced by a precision clutch

Real webcam use found the velocity-gain pointer jitterier than the pre-feature one, and pinch-to-click worst of all, while click-and-drag was better. Traced on seeded 30 FPS streams with uneven frame timing and .002-.004 normalized landmark noise, comparing against the pre-feature mapper extracted from git on identical input.

Root cause: the gain was carried as a persistent offset accumulated from the frame-to-frame step of the absolute target, which makes it an integrator of a differentiated noisy landmark. Measured consequences: the offset random-walked +-15 px over 20 s of a *stationary* hand; deviation from the absolute map averaged 77-110 px and peaked at 188 px during ordinary movement; and four different approaches to one hand position settled 178 px apart. The map had stopped being memoryless, which is a closed-loop failure no amount of smoothing repairs. Stationary high-frequency jitter was in fact slightly *lower* than legacy, which is why open-loop tests passed it -- and `test_the_offset_stays_bounded` had certified the 154 px lead as correct.

Replaced with an explicitly gated clutch: `target = gain*base + (1-gain)*anchor` while engaged, which is affine in the *current* absolute position, so nothing integrates. Results on the same traces: deviation from the absolute map is 0.00 px on fast and normal movement; the settled-position spread across six approach histories is 17 px against 178; 30 s of slow/fast cycling accumulates nothing (worst clutch offset equals its 15 px cap); stationary wobble improves 2.1x peak-to-peak and 2.9x RMS against legacy; and precision mode changed state at most twice over 8 s at every speed swept through its hysteresis band.

Also corrected in the drag estimator, from code-level evidence rather than feel: alpha .72 passed landmark noise (a still drag shimmered 10 px/frame against legacy's 4.6) while a fixed .30 lagged a fast drag 85 px and then locked out -- a rejected frame froze the velocity, which froze the outlier limit, which guaranteed further rejection, and the drag lost the ability to throw at all. Alpha is now chosen from the residual, beta from alpha, the outlier limit is driven by the measurement rather than the filter and widens on consecutive rejections, and a reacquisition contributes position only, never velocity. The still drag is now 1.8x steadier than legacy and tracking beats it at every speed from 900 to 15000 px/s.

Pinch-to-click: the pointing landmark is the fingertip, so closing a pinch moved it ~30 px, and one noisy frame then broke the 10 px drag deadzone and started the cursor following raw landmarks. Gated on pinch settling and on the middle knuckle instead. Cursor movement after mouse-down fell from 35.2 px of path and 17.3 px worst (legacy) to 4.1 px and 2.9 px; a stationary click starts a drag in 6 runs of 30 rather than 30 of 30; and a decisive drag still acquires in 49 ms against legacy's 33 ms, a gentle one in about 110 ms.

296 automated tests pass, including new `tests/traces.py` trace infrastructure, `tests/test_pointer.py` (32) and `tests/test_click.py` (18). The retired `tests/test_gain.py` encoded the offset model and was removed.

Not verified: interactive feel of any of it. Whether 0.10/0.18 units/s match a real hand's settle, whether 15 px of clutch range is useful, whether the 110 ms gentle-drag delay is noticeable, and whether pinch-to-click now feels stable all need a human at a webcam. No desktop pointer was moved.
