import os
from pathlib import Path

from configs.llama_3_3b.lens_token_identity import sweep
from plotting.lens_overall import build_plot

OUTPUT_DIR = Path("artifacts")
os.makedirs(OUTPUT_DIR / "token_identity", exist_ok=True)

build_plot(sweep=sweep, correctness=True).save(OUTPUT_DIR / "token_identity" / "lens_correct.pdf")
build_plot(sweep=sweep, correctness=False).save(OUTPUT_DIR / "token_identity" / "lens_incorrect.pdf")
