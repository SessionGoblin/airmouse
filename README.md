# AirMouse

A local-only webcam mouse for Linux, with a Qt desktop interface, MediaPipe hand tracking, calibrated absolute positioning, adaptive smoothing, pinch click/drag, right click, and deliberate scrolling. Starts paused; preview works even when desktop input permissions are unavailable.

## Install and launch

Python 3.11+ and a webcam are required. Tested here with Python 3.12.14 on Linux. On Debian/Ubuntu, a normal system-Python installation needs these build/runtime packages:

```sh
sudo apt install python3-venv python3-dev build-essential linux-libc-dev libgl1 libegl1 libxcb-cursor0
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
python -m airmouse --download-model
python -m airmouse --debug
```

Run these commands from the repository directory. The model setup downloads Google's versioned hand landmark model (~8 MB) to `~/.cache/airmouse/hand_landmarker.task`. Camera processing thereafter uses no network connection. No frames are saved or uploaded. An offline installation can copy that model file into the same location.

The repository already has a working `.venv` on the development host. Launch directly with `.venv/bin/airmouse --debug`. The existing environment was built with the bundled Python runtime and `CC=gcc LDSHARED='gcc -shared'` because the host's system Python lacks development headers.

## First use and calibration

1. Select a webcam, then **Start preview**. Some Linux video nodes carry metadata instead of images; choose another camera index if opening fails.
2. Open **Calibrate active region & gestures**. The symmetric inner region defaults to an 18% margin, allowing screen edges to be reached without leaving the camera image. Increase the margin if your comfortable range is smaller. Apply settings to update the preview region.
3. Adjust sensitivity (1–2), smoothing time, dead zone, pinch/release ratios, confirmation delay, and cooldown. Ratios are relative to wrist-to-middle-knuckle palm length and account for frame aspect ratio. Release must exceed pinch to provide hysteresis.
4. Toggle any gestures you do not want. Changing settings pauses control. Calibration blocks F8 resumption until closed and leaves control paused afterward.
5. Explicitly select **Enable control** or press **F8**. Hold an unpinched pointing hand steady for the neutral dwell (default 350 ms), then move. Entering the camera while already pinching never presses a button.

Use good lighting, a visible palm, and one hand. Hand selection is limited to one detected hand; avoid placing a second hand in the active region. Higher smoothing suppresses jitter but increases lag. The configurable dead zone suppresses small stationary movements; the adaptive EMA responds faster for large movements. A speed limit prevents abrupt acquisition jumps. Screen topology is read at launch; restart after rearranging monitors. Absolute mapping covers the desktop bounding rectangle; gaps between irregularly arranged displays are resolved by the desktop compositor.

## Gestures and safety

Precision improvements: cursor filtering retains subpixel motion, so slow aiming is no longer rounded away each frame. Closing a pinch anchors the drag at the existing cursor position. The cursor stays still until movement exceeds **Drag start distance** (10 screen pixels by default, adjustable in calibration); subsequent drag motion is relative to that anchor. Increase this distance if clicks accidentally become drags.

Holding a right pinch freezes the cursor until release. Scrolling now requires straight index/middle fingers and clearly bent ring/little fingers, and tolerates 120 ms of pose flicker without moving the pointer. Nonfinite detections and abrupt quarter-frame jumps within 120 ms are discarded and require neutral reacquisition; this can briefly interrupt exceptionally fast hand motions. Reacquisition starts from the last cursor position instead of the screen center.

| Action | Gesture / key |
| --- | --- |
| Move | Index fingertip in the green active region |
| Left click | Thumb-index pinch, then release |
| Drag | Hold the same pinch while moving; release to drop |
| Right click | Thumb-middle pinch; open before another right click |
| Scroll | Index and middle extended, ring and little fingers folded; hold 350 ms, then move vertically |
| Pause / resume | Global **F8** |
| Emergency stop | Global **F12**; always pauses and releases the left button |
| Local stop | **Escape** when the app is focused, or the red stop button |

Pinches require continuous confirmation (80 ms default). Pointer motion freezes while a gesture is being confirmed. Right click is latched until the hand opens; clicks have a 350 ms cooldown. Scrolling locks pointer movement. Loss of tracking immediately releases a drag, discards gesture state, and requires neutral reacquisition. Stalled vision (>300 ms) pauses control via the UI watchdog. Camera stop, settings changes, hotkey listener failure, and application exit also release the button. F12 releases from the keyboard listener thread independently of inference; it does not rely on receiving another camera frame.

Releasing a held drag on tracking loss can drop the item at its current location. Resume is always explicit after emergency stop. A global keyboard listener is required before control can be enabled.

## Linux input support

**X11:** uses `pynput` for desktop input and global F8/F12. Launch inside the graphical session with access to its display. XWayland is not treated as full Wayland support.

**Wayland:** uses a Linux `uinput` absolute pointer and reads F8/F12 from accessible physical keyboard event devices. It does not fall back to XWayland injection, which cannot control all native Wayland applications. If the necessary devices are inaccessible, a clear **Preview only** message appears and Enable control remains disabled. The current development host has this restriction.

A system administrator can provide scoped device ACLs for the current session, for example:

```sh
sudo modprobe uinput
# Replace eventN with your physical keyboard, identified under /dev/input/by-id/.
sudo setfacl -m u:"$(id -un)":rw /dev/uinput
sudo setfacl -m u:"$(id -un)":r /dev/input/eventN
```

These are manual setup examples; the app never changes permissions or runs as root. Keyboard-device read access exposes keyboard events to processes running as your user, so grant access only to the intended keyboard rather than all input devices. ACLs may need reapplying after reboot or reconnect. Restart preview after granting access. Revoke with `sudo setfacl -x u:"$(id -un)" /dev/uinput /dev/input/eventN`.

Compositors differ in virtual absolute-pointer handling and display assignment. Wayland pointer injection and global keyboard actions could not be tested on this host because its device permissions were restricted. Verify edge reachability and F12 on your compositor before ordinary use. On reconnecting a keyboard, restart preview to discover it again.

Windows and macOS have an isolated `pynput` adapter, but are not validated targets of this Linux-first release. macOS may require Accessibility and Input Monitoring permissions. Mixed-DPI multi-monitor setups also need platform validation.

## Debugging and tests

```sh
pytest -q
python scripts/smoke.py
python scripts/smoke.py --camera 0
```

The unit and offscreen UI tests do not move the desktop pointer. The optional camera smoke test processes local frames for four seconds and verifies cleanup. `--debug` shows landmark indices and coordinates, normalized pinch distances, index/middle joint angles, state transitions, processing FPS, and the injected cursor target. The confidence label reports **handedness classification confidence**; MediaPipe does not provide per-frame tracking confidence through this result, so tracking itself is shown as present/absent.

Settings persist atomically in `~/.config/airmouse/settings.json`; invalid files fall back to safe defaults. Delete this file to reset calibration.

Troubleshooting:

- **Model missing:** run the explicit model download command above. The app never downloads while using the webcam.
- **Cannot open webcam:** close other camera apps, verify OS camera permission and video-device access, or select a different index.
- **Preview only:** read the input status message and Linux permission instructions above.
- **Qt platform plugin error:** install the display runtime packages listed above; `QT_QPA_PLATFORM=wayland` can select native Wayland. `QT_QPA_PLATFORM=offscreen` is only for tests.
- **Python.h / compiler missing:** install the matching Python development package and build tools before pip installation. Bundled Python builds may need `CC=gcc LDSHARED='gcc -shared'`.
- **Low FPS:** close other camera users, use good lighting, and keep the hand visible. Capture uses a single latest-frame slot at a requested 640×480 / 30 FPS, preventing a queue of old frames.
- **Clicks too easy:** decrease pinch ratio, increase confirmation time, or disable the gesture.
- **Scrolling accidentally:** increase neutral/scroll dwell or disable scrolling.

## Design

- `capture.py`: independent capture thread and latest-frame mailbox.
- `tracking.py`: MediaPipe VIDEO-mode detector with monotonic timestamps.
- `gestures.py`: pure feature extraction and explicit idle/pointing/pinch-down/dragging/release/right-click/scrolling/paused state machine.
- `mapping.py`: calibrated screen mapping and replaceable adaptive EMA filter.
- `input.py`: input protocol, X11/native and Wayland adapters, global hotkeys.
- `controller.py`: locked safety gate, state-to-action dispatch, speed limiting.
- `worker.py`: inference worker and resource lifecycle.
- `config.py`, `calibration.py`, `ui.py`: validation, persistence, calibration, and desktop UI.

Upper-body commands are a deferred stretch goal and are not implemented. They can be added as a separate recognizer without changing the primary hand-to-pointer path.

## References

- [MediaPipe HandLandmarker API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker)
- [python-evdev documentation](https://python-evdev.readthedocs.io/en/stable/)
- [pynput platform limitations](https://pynput.readthedocs.io/en/latest/limitations.html)
