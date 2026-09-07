#!/usr/bin/env python3
"""Convert released OPA-DPO 7B rollouts to the official TARS parquet schema."""
import argparse, base64, glob, hashlib, json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def repeats_last_sentence(text):
    parts=text.split('.')
    return len(parts)>=2 and bool(parts[-2].strip()) and parts[-2].strip() in '.'.join(parts[:-2])


def repeats_last_word(text):
    words=text.split()
    return len(words)>=2 and words[:-2].count(words[-1].strip())>30


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rollouts-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--manifest',required=True)
    a=p.parse_args()
    files=sorted(glob.glob(str(Path(a.rollouts_root)/'llava7b_online_generation_subset*'/'rollouts'/'*.json')))
    if not files: raise FileNotFoundError(f'No LLaVA-7B rollout JSON under {a.rollouts_root}')
    schema=pa.schema([
      ('ds_name',pa.string()),('image',pa.struct([('bytes',pa.binary()),('path',pa.string())])),
      ('text',pa.string()),('origin_dataset',pa.string()),('origin_split',pa.string()),
      ('idx',pa.int64()),('image_path',pa.string()),
      ('win_similarity_score',pa.list_(pa.float32())),('rej_similarity_score',pa.list_(pa.float32()))])
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_suffix('.parquet.partial')
    writer=pq.ParquetWriter(tmp,schema,compression='none'); batch=[]; raw=kept=identical=0
    removed={'empty_report':0,'terminal_repetition':0,'empty_correction':0}
    def flush():
        nonlocal batch
        if batch: writer.write_table(pa.Table.from_pylist(batch,schema=schema)); batch=[]
    try:
        for f in files:
            for x in json.load(open(f,encoding='utf-8')):
                raw+=1
                if json.dumps(x.get('AI_json_report',''),ensure_ascii=False,indent=4)=='""': removed['empty_report']+=1; continue
                rejected=x.get('original_generate_response','') or ''
                if repeats_last_sentence(rejected) or repeats_last_word(rejected): removed['terminal_repetition']+=1; continue
                chosen=x.get('AI_pseudo_response','') or ''
                if not isinstance(chosen,str) or not chosen: removed['empty_correction']+=1; continue
                q=(x.get('query','') or '').replace('<image>','').strip(); image=base64.b64decode(x['image_bytes'])
                identical += chosen.strip()==rejected.strip()
                batch.append({'ds_name':'OPA-DPO-7B','image':{'bytes':image,'path':x.get('image_id',f'opa_{kept}.jpg')},
                  'text':json.dumps({'question':q,'chosen':chosen,'rejected':rejected},ensure_ascii=False),
                  'origin_dataset':'RLAIF-V/OPA-DPO','origin_split':'llava7b_subset1+subset2','idx':kept,
                  'image_path':x.get('image_id',f'opa_{kept}.jpg'),'win_similarity_score':[],'rej_similarity_score':[]})
                kept+=1
                if len(batch)>=64: flush()
        flush(); writer.close(); tmp.replace(out)
    except BaseException:
        writer.close()
        raise
    manifest={'source_files':len(files),'raw_rows':raw,'retained_rows':kept,'removed':removed,
      'identical_pairs_retained':identical,
      'preference_mapping':{'chosen':'AI_pseudo_response (GPT-4V correction)','rejected':'original_generate_response (LLaVA-1.5-7B)'},
      'output':str(out.resolve()),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    Path(a.manifest).write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
