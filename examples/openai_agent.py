import sys

from locus_eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", "--agent", "openai", *sys.argv[1:]]))
