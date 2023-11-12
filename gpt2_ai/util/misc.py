import subprocess
from pathlib import Path


def get_git_root() -> Path:
    try:
        # Run the 'git rev-parse --show-toplevel' command to get the root directory
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())
        else:
            # Handle the case where the command failed (e.g., not in a Git repository)
            raise RuntimeError(
                f"git rev-parse --show-toplevel failed with return code",
                f"{result.returncode}. Probably not in a Git repository.")
    except FileNotFoundError:
        # Handle the case where Git is not installed or not in the system PATH
        return RuntimeError("Git is not installed or not in the system PATH.")

def get_whatever():
    out = subprocess.run(['dip', 'rev-parse', '--show-toplevel'])
    print(out)

if __name__ == "__main__":
    out = get_git_root()
    print(out)