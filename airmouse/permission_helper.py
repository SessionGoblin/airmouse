"""Small standalone helper run through pkexec after explicit user confirmation."""
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys


def keyboards():
    result = []
    for entry in sorted(Path('/sys/class/input').glob('event*')):
        try:
            if '/virtual/' in str(entry.resolve()):
                continue
            words = (entry/'device/capabilities/key').read_text().split()
            bits = sum(int(word, 16) << (i * struct.calcsize('L') * 8)
                       for i, word in enumerate(reversed(words)))
            # Linux KEY_F8=66 and KEY_F12=88; only physical keyboards with both.
            if bits & (1 << 66) and bits & (1 << 88):
                result.append(('/dev/input/' + entry.name, (entry/'device/name').read_text().strip()))
        except (OSError, ValueError):
            continue
    return result


def grant(keyboard):
    uid = os.environ.get('PKEXEC_UID', '')
    if os.geteuid() != 0 or not uid.isdecimal() or int(uid) == 0:
        raise RuntimeError('Run this helper through the AirMouse permission dialog.')
    if not re.fullmatch(r'/dev/input/event\d+', keyboard) or keyboard not in dict(keyboards()):
        raise RuntimeError('The selected physical keyboard is no longer available. Retry setup.')
    if not stat.S_ISCHR(os.lstat(keyboard).st_mode):
        raise RuntimeError('The selected keyboard is not a device.')
    subprocess.run(['/usr/sbin/modprobe', 'uinput'], check=True)
    if not stat.S_ISCHR(os.lstat('/dev/uinput').st_mode):
        raise RuntimeError('The virtual pointer device is unavailable.')
    # Scope access to the authenticated caller and the selected devices only.
    subprocess.run(['/usr/bin/setfacl', '-m', f'u:{int(uid)}:rw', '/dev/uinput'], check=True)
    subprocess.run(['/usr/bin/setfacl', '-m', f'u:{int(uid)}:r', keyboard], check=True)


if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            raise RuntimeError('Expected one physical keyboard.')
        grant(sys.argv[1])
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
