import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    wheel = parser.parse_args().wheel.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="locus-wheel-") as temporary:
        root = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(root / "environment")
        python = root / "environment" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        console = (
            root
            / "environment"
            / ("Scripts/locus-eval.exe" if os.name == "nt" else "bin/locus-eval")
        )
        commands = (
            [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            [str(console), "demo", "--output", "report"],
            [str(python), "-m", "locus_eval", "replay", "report"],
        )
        for command in commands:
            subprocess.run(command, cwd=root, check=True, timeout=120)


if __name__ == "__main__":
    main()
