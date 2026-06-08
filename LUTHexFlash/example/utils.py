import subprocess
import sys


def _run(cmd, cwd=None):
    subprocess.run(cmd, stdout=sys.stdout, cwd=cwd, check=True)
