import os
from pathlib import Path

from configs.llama_3_3b.lens_token_identity import sweep
from plotting.lens_tasks import build_chart_per_task

OUTPUT_DIR = Path("artifacts")


os.makedirs(OUTPUT_DIR / "token_identity" / "lens", exist_ok=True)
for e in sweep.experiments:
    correct = "correct" if e.correctness else "incorrect"
    if len(e.step_result("processing_signatures")) >= 10:
        build_chart_per_task(e).save(OUTPUT_DIR / "token_identity" / "lens" / f"{e.task_name}_{correct}.pdf")


os.makedirs(OUTPUT_DIR / "token_identity" / "lens_excluded", exist_ok=True)
for e in sweep.experiments:
    correct = "correct" if e.correctness else "incorrect"
    if 0 < len(e.step_result("processing_signatures")) < 10:
        build_chart_per_task(e).save(OUTPUT_DIR / "token_identity" / "lens_excluded" / f"{e.task_name}_{correct}.pdf")
