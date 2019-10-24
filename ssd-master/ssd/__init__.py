import sys
import subprocess
sys.path.append(subprocess.check_output(
    ['git', 'rev-parse', '--show-toplevel']
    ).strip().decode("utf-8"))
