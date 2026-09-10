# AirMouse

https://github.com/user-attachments/assets/436311e0-316f-4552-9c9a-95ba83f2c343


A local-only webcam mouse for Linux, with a Qt desktop interface, MediaPipe hand tracking, calibrated absolute positioning, adaptive smoothing, pinch click/drag, right click, and deliberate scrolling. Starts paused; preview works even when desktop input permissions are unavailable. You are able to record gestures and positional hand movement paths to assign to actions, including arbitrary commands.

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

1. Select a webcam and **Resolution**, then **Start preview**. Resolution defaults to 640 × 480 and is saved automatically. Stop preview before changing it. The status line shows the actual frame size and, if different, the requested size; available presets are requests, not a list of modes guaranteed by your camera. Capture requests MJPG so higher resolutions still negotiate 30 FPS where the camera supports it; **Hold frame rate in low light** (on by default) asks the webcam not to drop below that for brightness, and restores the previous control when preview stops. Frames above 1280 px wide are scaled down before tracking, since the extra pixels do not improve landmark accuracy. Some Linux video nodes carry metadata instead of images; choose another camera index if opening fails.
2. Open **Calibrate active region & gestures**. The symmetric inner region defaults to an 18% margin, allowing screen edges to be reached without leaving the camera image. Increase the margin if your comfortable range is smaller. Apply settings to update the preview region.
3. Adjust sensitivity (1–2), smoothing time, dead zone, pinch/release ratios, confirmation delay, and cooldown. Ratios are relative to wrist-to-middle-knuckle palm length and account for frame aspect ratio. Release must exceed pinch to provide hysteresis, and an established drag gets wider hysteresis again.
4. Toggle any gestures you do not want. Changing settings pauses control. Calibration blocks F8 resumption until closed and leaves control paused afterward.
5. Explicitly select **Enable control** or press **F8**. Hold an unpinched pointing hand steady for the neutral dwell (default 350 ms), then move. Entering the camera while already pinching never presses a button.

Use good lighting, a visible palm, and one hand. Hand selection is limited to one detected hand; avoid placing a second hand in the active region. Higher smoothing suppresses jitter but increases lag. The configurable dead zone suppresses small stationary movements; the adaptive EMA responds faster for large movements. A speed limit prevents abrupt acquisition jumps. Screen topology is read at launch; restart after rearranging monitors. Absolute mapping covers the desktop bounding rectangle. Targets in gaps between irregularly arranged displays snap to the nearest visible monitor edge, for both pointing and dragging. The internal motion continues across gaps so the pointer can reach the next screen; the visible transition across a gap can exceed the normal movement speed limit. Wayland output assignment still depends on the compositor.

## Gestures and safety

Precision improvements: cursor filtering retains subpixel motion, so slow aiming is no longer rounded away each frame. Closing a pinch anchors the drag at the existing cursor position. The cursor stays still until movement exceeds **Drag start distance** (10 screen pixels by default, adjustable in calibration); subsequent drag motion is relative to that anchor. Increase this distance if clicks accidentally become drags.

**Low-speed precision assist** (on by default) is a clutch, not an acceleration curve. Ordinary and fast movement use the plain absolute map, unchanged and bit-for-bit identical to the pre-feature pointer: there is no speed at which the map stops being memoryless, so the same hand position always means the same place on screen. When your hand settles — below about 0.10 units/s for 120 ms, releasing again above 0.18 — the cursor anchors and then moves half as far for the same hand movement, over a short local range capped at 0.8% of the widest screen span (about 15 px on a 1920 desktop). Beyond that the hand tracks the map one-to-one again, and the offset bleeds to zero as soon as you move normally. Most of the benefit is not the gain at all but the extra smoothing that comes with it: on 30 FPS traces with realistic landmark noise the cursor's stationary wobble drops from about 18 px peak-to-peak to 8, which is what actually decides whether you can hold on a small target. Turn the checkbox off for exactly the pre-feature behaviour.

An earlier version of this did use a velocity-dependent gain curve, layered over the absolute map as a persistent offset. Do not reintroduce that shape: because the offset was accumulated from the frame-to-frame step of a noisy landmark, it was an integrator, and it made the cursor's position depend on the route the hand took to get there — four approaches to one point settled up to 178 px apart. The pointer stopped being learnable, which no amount of smoothing can repair.

**Click stability.** The pointing landmark is the index fingertip, and closing a pinch swings it by roughly 30 px of mapped travel while the hand itself stays put — so the cursor drifts off the target during the gesture meant to commit to it. Two gates stop that. A press does not allow a drag until the pinch has stopped closing (a rate test, with a 120 ms fallback), so the last stretch of finger movement is not read as hand movement. And whether the hand has really moved is judged on the middle knuckle rather than the fingertip, low-passed and required to clear the deadzone on four consecutive frames — because on a 1920 desktop the mapping is about 3000 px per unit of hand travel, so even mild landmark noise is 6–9 px against a 10 px **Drag start distance**, and a raw per-frame comparison turned essentially every click into a short drag. A displacement 2.5× the deadzone bypasses the wait, so a decisive drag still starts within a frame or two.

**Drag persistence.** Starting a drag still needs a firm pinch, but keeping one does not: an established drag exits only on a pinch that has clearly opened (35% wider than **Release**), rides out frames the detector cannot read at all for a `drag_grace` period (150 ms by default), and estimates the anchor from its recent trajectory while the pose is uncertain — leaning on the measurement when the pinch is firm and on the trajectory when it is not, then converging back rather than snapping. Implausible single-frame landmark excursions are clamped to a rate a hand could actually move at instead of being followed. Stability here comes from rejecting bad measurements rather than from heavier smoothing, so a dragged window stays tightly coupled to the hand.

Holding a right pinch freezes the cursor until release. Scrolling now requires straight index/middle fingers and clearly bent ring/little fingers, and tolerates 120 ms of pose flicker without moving the pointer. Nonfinite detections and abrupt quarter-frame jumps within 120 ms are discarded and require neutral reacquisition; this can briefly interrupt exceptionally fast hand motions. Reacquisition starts from the last cursor position instead of the screen center.

| Action | Gesture / key |
| --- | --- |
| Move | Index fingertip in the green active region |
| Left click | Thumb-index pinch, then release |
| Drag | Hold the same pinch while moving; release to drop |
| Right click | Thumb-middle pinch; open before another right click |
| Scroll | Index and middle extended, ring and little fingers folded; hold 350 ms, then move vertically |
| Custom pose | Any recorded hand shape, held 450 ms; see below |
| Fling | Release a drag while still moving; the window coasts and settles |
| Pause / resume | Global **F8** |
| Emergency stop | Global **F12**; always pauses and releases the left button |
| Local stop | **Escape** when the app is focused, or the red stop button |

Pinches require continuous confirmation (80 ms default). Pointer motion freezes while a gesture is being confirmed. Right click is latched until the hand opens; clicks have a 350 ms cooldown. Scrolling locks pointer movement. Loss of tracking releases a drag once the hand has been unreadable for longer than the grace period, then discards gesture state and requires neutral reacquisition; a lost hand can never throw, because only an opened pinch is treated as letting go. An emergency stop takes no grace period and releases on the spot. Stalled vision (>300 ms) pauses control via the UI watchdog. Camera stop, settings changes, hotkey listener failure, and application exit also release the button. F12 releases from the keyboard listener thread independently of inference; it does not rely on receiving another camera frame.

#### Two hands

**Track a second hand** adds a modifier hand alongside the pointer. Roles are shown in the preview — the pointer hand is drawn in amber with a crosshair on its index fingertip, the modifier in violet — so a role changing hands is visible directly rather than only as the cursor jumping.

**Pointer** chooses how roles are decided. **Right** or **Left** pins the pointer to that side of the mirrored preview and recomputes it every frame, so roles never drift — crossing your hands swaps them, predictably, and crossing back puts them right. Invert it to swap which hand points. **Automatic** follows each hand through a crossing instead, which is better when you cross often but can settle the wrong way round and stay there. Right is the default.

Under Automatic, roles are assigned by position and motion, never by the model's handedness label: that classifier flickers exactly when two hands are close or overlapping, which is when a swap would be most damaging. Each hand is matched to where its role is predicted to be, so hands reaching across each other keep their roles through the crossing; two hands crossing at mirrored speeds are momentarily coincident and genuinely ambiguous. A role keeps its slot for 0.4 s after its hand leaves, so a brief dropout does not reshuffle. A lone hand always takes the pointer. **Pointer** selects which side seeds the cursor when both hands first appear — position in the mirrored preview, so your right hand is on the right whatever the label says.

The modifier hand gates drawn gestures and carries its own poses (below). Costs frame rate: with two hands enabled the model keeps hunting for a second hand whenever only one is visible, so watch the FPS in the status line and turn it off if it hurts.

### Modifier poses and modes

A recorded pose can be read from either hand. **Hand** in the gesture's own settings chooses which — it sets the hand a new recording samples from, and can be changed on an existing gesture at any time. No re-recording is needed: poses are stored mirrored into a single chirality, so the recording is already valid for either hand and only the hand it is read from changes.

A held modifier pose is also a **mode**, and that is what makes it a modifier rather than a second pose slot. Pointer poses and drawn strokes can be scoped with **Only while** / **Only under**, so three modifier poses multiply the gestures you already have instead of adding three more to remember. Unscoped gestures keep working in every mode — holding a modifier never switches off your ordinary gestures — and a scoped gesture outranks an unscoped one of the same shape, since the more specific binding is the one you raised the modifier for.

The gate uses the same pinch/release hysteresis as clicking, so pinch noise cannot chop one stroke into several too-short ones.

Mark a modifier pose **Holding this opens stroke drawing** to replace the built-in modifier pinch. Once any gate pose exists the pinch no longer opens drawing, and holding a gate freezes the cursor — `--debug` reports which gate is in use and whether it is currently open, so a gate engaging unexpectedly is visible rather than looking like the pointer has locked up. Strokes can then be scoped per gate, so the same shape drawn under two different modifier poses means two different things. A gate pose does not also run its own binding; it is a mode, not an event.

A chord costs two confirmations in sequence — the modifier must settle before the scoped pose becomes eligible — so it takes about twice **Custom pose confirmation** to fire. That is the price of not triggering chords on a modifier shape passing through on its way somewhere else; lower `custom_dwell` in calibration if it feels slow. The current mode is shown in the status line and in `--debug`.

Because roles select the hand, the same shape can mean one thing on the pointer and another on the modifier without conflicting, and poses stay hand-agnostic — role assignment already picks the hand by position, so locking the mirror as well would only break things when you raise your hands in the other order.

## Drawn gestures

**Drawn gestures** records a shape traced with your pointer index finger and binds it to the same actions a pose can trigger. Pinch your modifier hand to start drawing, trace, then open the pinch to finish; the cursor is frozen for the duration, and a drag in progress is released rather than smeared along the stroke.

The modifier pinch is what makes this workable. Because the pointer *is* your hand here, a recognizer running freely over the cursor path would fire during ordinary pointing — knowing where a stroke starts and ends is the hard half of path recognition, and the second hand supplies it for free instead of costing a gesture.

Matching uses the $1 unistroke recognizer: the path is resampled to 64 evenly spaced points, so drawing speed does not matter, then scaled and compared under a small rotation search. **Direction is preserved by default** — classic $1 normalizes every stroke to its own heading, which would make a left swipe, a right swipe and an up swipe the same straight line. **Match at any orientation** opts a stroke out when the shape matters and the angle does not, at the cost of merging strokes that differ only in direction. Measured against synthetic strokes: the same shape redrawn scores .94–.99, perpendicular directions .56, opposite directions .29, and an unrelated scribble .39, against an accept threshold of .78. Strokes are stored in `~/.config/airmouse/strokes.json`, and the recorder warns when a new stroke scores as high against an existing one as a real match would.

## Custom poses

The gesture dialogs are not modal: the main window stays visible and its preview keeps updating while you record, since holding a pose where the camera can see it is hard to do behind a window that hides the camera view. Control stays paused for as long as a dialog is open.

**Custom gestures** records a hand shape and binds it to a keyboard shortcut, an AirMouse command (pause, pause/resume, recentre pointer), or a shell command. Hold the pose for three seconds of countdown plus about a second of sampling; frames captured while the hand is still moving are discarded, and a pose that never settles is refused rather than saved as a template that would match nothing.

A pose is stored relative to the wrist, scaled by the wrist-to-knuckle span, rotated upright and mirrored into a single chirality, so it matches wherever your hand is in frame, however far from the camera, at any tilt, and **with either hand**. Two checkboxes narrow that when you want the distinction: **Only match at the tilt it was recorded at** separates poses that are the same shape rotated, such as thumbs up and thumbs down, and **Only match with the hand it was recorded with** gives each hand its own gesture. The hand check reads the winding of the palm triangle, so it needs to see your palm; held edge-on the winding collapses and the hand becomes unknown, at which point a hand-locked pose stops matching while an either-hand pose keeps working.

Templates are matched before the built-in pinches, so a pose that also reads as a pinch will shadow clicking while it is held; the recorder warns when it detects this, and when a new pose is too close to an existing one to ever win a match. Custom poses fire once per hold, are confirmed over 450 ms (longer than a pinch, because a binding can run a command), and share the standard action cooldown. Shell commands are run without a shell, so they cannot expand globs or chain operators. The virtual keyboard behind a shortcut binding is built in the background as soon as something is bound to one, because creating it waits on udev and doing that while a gesture fires would stall tracking; if a shortcut is bound mid-session the first press may be dropped while the device comes up, and the next one lands. **F8** and **F12** cannot be bound; the app's own hotkey listener would read them back. Templates are stored in `~/.config/airmouse/gestures.json`.

Releasing a held drag on tracking loss drops the item at its current location, after the grace period and without a throw. Resume is always explicit after emergency stop. A global keyboard listener is required before control can be enabled.

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

The unit and offscreen UI tests do not move the desktop pointer. The optional camera smoke test processes local frames for four seconds and verifies cleanup. `--debug` shows landmark indices and coordinates, normalized pinch distances, index/middle joint angles, state transitions, processing FPS, and the injected cursor target. It also reports instantaneous and robust hand speed with the cursor ratio they produce, whether the precision clutch is engaged and how far it is displacing the cursor, the deviation of the emitted pointer from the plain absolute map (which must be zero outside precision mode -- anything else means the map has stopped being memoryless), whether a press has finished closing and how far the knuckle has moved since it, the pinch confidence and drag tracking quality, the measured against the estimated drag anchor, how long a latched drag has been running blind, and why the last drag ended together with its release velocity and whether a throw was accepted or refused -- none of which is otherwise visible, since gain and a latched drag both change what the pointer did without changing the state name. Nothing is printed; the readout is built only while the pane is open. The confidence label reports **handedness classification confidence**; MediaPipe does not provide per-frame tracking confidence through this result, so tracking itself is shown as present/absent.

Settings persist atomically in `~/.config/airmouse/settings.json`; invalid files fall back to safe defaults. Delete this file to reset calibration.

Troubleshooting:

- **Model missing:** run the explicit model download command above. The app never downloads while using the webcam.
- **Cannot open webcam:** close other camera apps, verify OS camera permission and video-device access, or select a different index.
- **Preview only:** read the input status message and Linux permission instructions above.
- **Qt platform plugin error:** install the display runtime packages listed above; `QT_QPA_PLATFORM=wayland` can select native Wayland. `QT_QPA_PLATFORM=offscreen` is only for tests.
- **Python.h / compiler missing:** install the matching Python development package and build tools before pip installation. Bundled Python builds may need `CC=gcc LDSHARED='gcc -shared'`.
- **Low FPS:** close other camera users and keep the hand visible. Capture requests MJPG at 30 FPS. If the picture is 15 FPS and dim, leave **Hold frame rate in low light** on (the preview may look darker). Turn it off if you would rather keep auto-exposure’s brighter, slower shutter. Cameras without MJPG may still be stuck at 5–10 FPS uncompressed YUYV at 720p/1080p; try 640×480 then.
- **Clicks too easy:** decrease pinch ratio, increase confirmation time, or disable the gesture.
- **Scrolling accidentally:** increase neutral/scroll dwell or disable scrolling.

## Design

- `capture.py`: independent capture thread and latest-frame mailbox.
- `tracking.py`: MediaPipe VIDEO-mode detector with monotonic timestamps.
- `gestures.py`: pure feature extraction and explicit idle/pointing/pinch-down/dragging/release/right-click/scrolling/paused state machine.
- `mapping.py`: calibrated screen mapping and every pointer filter -- adaptive EMA for pointing, an anchored precision clutch over it, and an alpha-beta drag-anchor estimator, each with its own tunable profile.
- `telemetry.py`: per-frame record of what the control loop decided, for the tuning diagnostics pane.
- `input.py`: input protocol, X11/native and Wayland adapters, global hotkeys.
- `controller.py`: locked safety gate, state-to-action dispatch, speed limiting.
- `worker.py`: inference worker and resource lifecycle.
- `config.py`, `calibration.py`, `ui.py`: validation, persistence, calibration, and desktop UI.

Upper-body commands are a deferred stretch goal and are not implemented. They can be added as a separate recognizer without changing the primary hand-to-pointer path.

## References

- [MediaPipe HandLandmarker API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker)
- [python-evdev documentation](https://python-evdev.readthedocs.io/en/stable/)
- [pynput platform limitations](https://pynput.readthedocs.io/en/latest/limitations.html)
