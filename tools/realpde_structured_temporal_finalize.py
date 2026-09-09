#!/usr/bin/env python3
"""Recover aggregate evidence from completed evaluation CSVs."""
import argparse,csv,json
from pathlib import Path
def read(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument("--arm-dir",type=Path,required=True);p.add_argument("--arm",required=True);a=p.parse_args(); rows=[]
    for u in (2000,2500,3000):
        d=read(a.arm_dir/f"eval_{u:05d}"/"window_metrics.csv"); rows.append({"arm":a.arm,"update":u,**{m:sum(float(x[m]) for x in d)/len(d) for m in ("rel_l2","tke","mvpe")}})
    with (a.arm_dir/"aggregate_metrics.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.arm_dir/"runtime.json").write_text(json.dumps({"parameter_count":None,"added_parameter_count":0,"peak_gpu_memory_allocated":None,"peak_gpu_memory_reserved":None,"training_wall_seconds":None,"inference_latency":None,"adapter_output_norm":0.0,"adapter_parameter_norm":0.0,"recovered_from_completed_eval":True},indent=2))
    (a.arm_dir/"run_metadata.json").write_text(json.dumps({"status":"REVIEW_REQUIRED","arm":a.arm,"physical_batch":8,"gradient_accumulation":False,"locked_final_accessed":False,"full_data_accessed":False,"sps_accessed":False,"codabench_accessed":False,"recovered_from_completed_eval":True},indent=2))
if __name__=="__main__":main()
