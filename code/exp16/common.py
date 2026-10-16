# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/16_strong_recommender_tactile/common.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: Exp16 경로·seed와 해시·JSON 입출력·실험 이벤트 공통 함수를 제공한다.
# ============================================================
"""Isolated full-catalog experiment utilities. Never write outside ROOT."""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np

# 원본 실험 폴더를 기준으로 데이터·artifact·모델 cache 경로를 계산한다.
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
WORKSPACE = PROJECT.parent
ART = ROOT / 'artifacts'
DATA = ROOT / 'data'
SEED = 20260904
CLASSES = ['soft','firm','smooth','rough','non_elastic','elastic','thin','thick','flexible','stiff','warm','cool','spongy','crisp']
RELIABLE = ['smooth','rough','thin','thick','flexible','stiff','warm','cool']
CHECKPOINT = PROJECT / 'experiments/12_class_multilabel_fashionclip_ft/models/fashionclip_last2.pt'
CHECKPOINT_SHA = '073b542a9d6bd4450aa1f48596c756cc0812b1b5d914945e8f94f7c3fda1934a'
CLIP = WORKSPACE / '.cache/huggingface/hub/models--patrickjohncyh--fashion-clip/snapshots/7e3ba62ce16b379a1ab479346b66f192e76f51b7'

# 파일을 chunk 단위로 읽어 재현성 기록용 SHA-256을 계산한다.
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()

# Exp16 루트 밖으로 쓰지 않는 것을 확인한 뒤 JSON을 원자적으로 교체한다.
def save_json(path, value):
    path=Path(path)
    assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def load_json(path):
    return json.loads(Path(path).read_text())

# 단계별 상태를 로컬 JSONL에 남기고 환경변수가 있을 때만 알림을 보낸다.
def event(stage, status, **extra):
    ART.mkdir(parents=True,exist_ok=True)
    row={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,'status':status,**extra}
    with open(ART/'events.jsonl','a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)
    # Secret belongs in the launch environment, never source/checkpoints/reports.
    url=os.environ.get('DISCORD_WEBHOOK_URL')
    if url and status in ('started','complete','failed','blocked'):
        import urllib.request
        message=f'[Strong recommender] {stage}: {status}\n'+json.dumps(extra,ensure_ascii=False)[:1200]
        try:
            req=urllib.request.Request(url,data=json.dumps({'content':message,'allowed_mentions':{'parse':[]}}).encode(),headers={'Content-Type':'application/json','User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req,timeout=15) as r: code=r.status
            with open(ART/'notifications.jsonl','a') as f: f.write(json.dumps({'time_utc':row['time_utc'],'stage':stage,'status':status,'http':code})+'\n')
        except Exception as e:
            # Exception text can include credentials. Record class only.
            with open(ART/'notifications.jsonl','a') as f: f.write(json.dumps({'stage':stage,'error_type':type(e).__name__})+'\n')

# interaction 수 같은 배열의 분위수와 임계값별 개수를 요약한다.
def distribution(a):
    a=np.asarray(a)
    return {'n':int(a.size),'mean':float(a.mean()) if a.size else None,'median':float(np.median(a)) if a.size else None,
            **{f'p{p}':float(np.percentile(a,p)) if a.size else None for p in (25,50,75,90,95)},
            **{f'ge_{k}':int((a>=k).sum()) for k in (1,2,3,5,10,20)},'zero':int((a==0).sum())}
