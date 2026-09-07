#!/usr/bin/env python3
"""Build OPA and OPA-DPO datasets from RLHF-V/TARS preference pairs."""
import argparse, base64, hashlib, json, random
from pathlib import Path

import pandas as pd
from datasets import Dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--parquet",required=True)
    p.add_argument("--output-root",required=True)
    p.add_argument("--num-samples",type=int,default=4800)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    df=pd.read_parquet(a.parquet)
    if a.num_samples and a.num_samples<len(df):
        # TARS does not publish its exact 4.8k indices. Preserve a deterministic manifest.
        ids=list(range(len(df))); random.Random(a.seed).shuffle(ids); df=df.iloc[ids[:a.num_samples]]
    rows=[]
    for source_index,(_,r) in enumerate(df.iterrows()):
        text=json.loads(r["text"])
        image=r["image"]; raw=image.get("bytes") if isinstance(image,dict) else None
        if not raw: raise ValueError(f"row {source_index} has no embedded image bytes")
        chosen=text["chosen"].strip(); rejected=text["rejected"].strip()
        if not chosen or not rejected: continue
        rows.append({"queries":"<image>\n"+text["question"].strip(),
          "image_bytes":base64.b64encode(raw).decode("ascii"),
          # OPA SFT expects reference + expert revision; TARS has one positive.
          "standard_response":chosen,"AI_pseudo_response":chosen,
          "original_generate_response":rejected,"AI_json_report":"{}",
          "source_index":int(r.get("idx",source_index))})
    root=Path(a.output_root); root.mkdir(parents=True,exist_ok=True)
    common={k:[x[k] for x in rows] for k in ("queries","image_bytes","standard_response","AI_pseudo_response","source_index")}
    Dataset.from_dict(common).save_to_disk(root/"opa_training_data-tars-7B")
    Dataset.from_dict({k:[x[k] for x in rows] for k in rows[0]}).save_to_disk(root/"opadpo_training_data-tars-7B")
    manifest={"source":str(Path(a.parquet).resolve()),"source_sha256":hashlib.sha256(Path(a.parquet).read_bytes()).hexdigest(),
      "selection_seed":a.seed,"requested_samples":a.num_samples,"retained_samples":len(rows),
      "mapping":{"chosen":["standard_response","AI_pseudo_response"],"rejected":"original_generate_response"},
      "limitations":["TARS exact 4.8k indices are unpublished; deterministic seed-42 sampling is used.",
       "RLHF-V has no GPT-4V token report; detailed-report/response-score/image-relation OPA losses must be disabled.",
       "standard_response and AI_pseudo_response duplicate the sole positive response during OPA SFT."]}
    (root/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
