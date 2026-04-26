#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    req = root / 'requirements.txt'
    cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(req)]
    print('Installing packages with:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    print('Done.')


if __name__ == '__main__':
    main()
