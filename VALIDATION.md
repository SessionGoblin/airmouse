# Validation record

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
