"""
Run an ablation study across multiple configurations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pcrort.multimodal.experiments.train_multimodal import run_experiment as run_multimodal
from pcrort.multimodal.experiments.train_radiology import run_experiment as run_radiology
from pcrort.multimodal.experiments.train_pathology import run_experiment as run_pathology
from pcrort.multimodal.experiments.train_genomics import run_experiment as run_genomics
from pcrort.multimodal.evaluation.plots import plot_ablation_study


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Study")
    parser.add_argument("--config", type=str, default="configs/multimodal/base.yaml", help="Base config")
    parser.add_argument("--output-dir", type=str, default="outputs/ablation", help="Ablation results dir")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {"name": "Rad_Only", "script": run_radiology, "config": "configs/multimodal/radiology.yaml", "modalities": "Radiology", "fusion": "None"},
        {"name": "Path_Only", "script": run_pathology, "config": "configs/multimodal/pathology.yaml", "modalities": "Pathology", "fusion": "None"},
        {"name": "Gen_Only", "script": run_genomics, "config": "configs/multimodal/genomics.yaml", "modalities": "Genomics", "fusion": "None"},
        {"name": "Rad_Path", "script": run_multimodal, "config": "configs/multimodal/multimodal.yaml", "modalities": "Radiology,Pathology", "fusion": "attention"},
        {"name": "Rad_Gen", "script": run_multimodal, "config": "configs/multimodal/multimodal.yaml", "modalities": "Radiology,Genomics", "fusion": "attention"},
        {"name": "Path_Gen", "script": run_multimodal, "config": "configs/multimodal/multimodal.yaml", "modalities": "Pathology,Genomics", "fusion": "attention"},
        {"name": "All_Concat", "script": run_multimodal, "config": "configs/multimodal/multimodal.yaml", "modalities": "Radiology,Pathology,Genomics", "fusion": "concat"},
        {"name": "All_Attention", "script": run_multimodal, "config": "configs/multimodal/multimodal.yaml", "modalities": "Radiology,Pathology,Genomics", "fusion": "attention"},
    ]

    results_list = []

    for exp in experiments:
        print(f"\\n{'='*50}\\nRunning Ablation: {exp['name']}\\n{'='*50}\\n")
        overrides = {
            "experiment": {
                "name": f"ablation_{exp['name']}",
                "smoke_test": args.smoke_test
            }
        }
        
        if exp["script"] == run_multimodal:
            mods = exp["modalities"].split(",")
            overrides["modalities"] = {
                "use_radiology": "Radiology" in mods,
                "use_pathology": "Pathology" in mods,
                "use_genomics": "Genomics" in mods,
            }
            overrides["fusion"] = {"method": exp["fusion"]}

        # Run experiment
        # It writes metrics to outputs/metrics by default, we intercept it or load from disk
        exp["script"](exp["config"], overrides)

        # Load metrics from disk
        metric_file = Path("outputs/metrics") / f"ablation_{exp['name']}_metrics.json"
        if metric_file.exists():
            with open(metric_file, "r") as f:
                metrics = json.load(f)
            
            row = {
                "model": exp["name"],
                "modalities": exp["modalities"],
                "fusion_method": exp["fusion"],
                "accuracy": metrics.get("test_accuracy", 0.0),
                "auroc": metrics.get("test_auroc", 0.0),
                "f1": metrics.get("test_f1", 0.0)
            }
            results_list.append(row)
        else:
            print(f"Warning: Metrics file not found for {exp['name']}")

    df = pd.DataFrame(results_list)
    df.to_csv(out_dir / "ablation_results.csv", index=False)
    
    # Plot
    plot_ablation_study(df, out_dir, ["png", "pdf"])
    print(f"\\nAblation study complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
