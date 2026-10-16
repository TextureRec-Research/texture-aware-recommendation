# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/16_strong_recommender_tactile/recommender.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: Exp16 공식 split을 읽어 추천 backbone을 학습하고 전체 카탈로그로 평가한다.
# ============================================================
import argparse
import pandas as pd
import torch
from common import *
from models import BPR,LightGCN,SASRec

# CPU thread, NumPy·PyTorch seed, TF32 사용 여부를 고정한다.
def setup():
    torch.set_num_threads(8);torch.manual_seed(SEED);np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# catalog와 공식 train/validation/test 이벤트를 사용자 이력 구조로 적재한다.
class RecData:
    def __init__(self):
        self.catalog=pd.read_parquet(DATA/'catalog.parquet')
        self.nitems=len(self.catalog)
        e=pd.read_parquet(DATA/'events.parquet')
        self.train=e[e.split=='train'].copy()
        self.validation=e[e.split=='validation'].copy()
        self.train_uids=np.sort(self.train.uid.unique())
        self.uid_to_train={int(u):j for j,u in enumerate(self.train_uids)}
        self.hist={int(u):g.iid.to_numpy() for u,g in self.train.groupby('uid',sort=False)}
        self.train['local_uid']=self.train.uid.map(self.uid_to_train)
        self.pop_order=np.lexsort((self.catalog.iid.to_numpy(),-self.catalog.train_count.to_numpy()))
        self.pop_rank=np.empty(self.nitems,dtype=np.int64);self.pop_rank[self.pop_order]=np.arange(1,self.nitems+1)
        # cold item은 train split에서 한 번도 양성으로 등장하지 않은 상품이다.
        self.cold=self.catalog.train_count.to_numpy()==0

    def histories(self,split):
        if split=='validation':return self.hist
        assert split=='test'
        h={u:seq.copy() for u,seq in self.hist.items()}
        for row in self.validation.itertuples():h[int(row.uid)]=np.append(h.get(int(row.uid),np.array([],dtype=np.int32)),row.iid)
        return h

# 설정의 모델 이름을 해당 backbone 클래스와 하이퍼파라미터에 연결한다.
def new_model(name,d,c):
    common={'nusers':len(d.train_uids),'nitems':d.nitems,'dim':c['dim']}
    if name=='bpr':return BPR(**common)
    if name=='lightgcn':
        edges=(torch.tensor(d.train.local_uid.to_numpy()),torch.tensor(d.train.iid.to_numpy()))
        return LightGCN(**common,edges=edges,layers=c['lightgcn_layers'])
    if name=='sasrec':return SASRec(**common,layers=c['sasrec_layers'],heads=c['sasrec_heads'],maxlen=c['sasrec_maxlen'],dropout=c['sasrec_dropout'])
    if name=='esasrec':
        from esasrec_adapter import ESASRec
        return ESASRec(**common,layers=c['sasrec_layers'],heads=c['sasrec_heads'],maxlen=c['sasrec_maxlen'],dropout=c['sasrec_dropout'])
    if name=='smore':
        from smore_adapter import make_smore
        return make_smore(d,c)
    raise ValueError(name)

# 사용자별 과거 상품을 오른쪽 정렬하고 0으로 padding한다.
def padded(hist,uids,maxlen):
    x=np.zeros((len(uids),maxlen),dtype=np.int64)
    for j,u in enumerate(uids):
        s=hist.get(int(u),[])
        if len(s):s=np.asarray(s[-maxlen:])+1;x[j,-len(s):]=s
    return x

# 하나의 정답 상품 rank에서 NDCG·HR·MRR·Recall을 계산한다.
def metrics(ranks):
    r=np.asarray(ranks);out={'users':len(r)}
    if not len(r):return {**out,**{k:None for k in ('ndcg_at_10','hr_at_10','mrr_at_10')}}
    for k in (5,10):
        ok=(r>0)&(r<=k)
        out[f'ndcg_at_{k}']=float(np.where(ok,1/np.log2(np.maximum(r,1)+1),0).mean())
        out[f'hr_at_{k}']=float(ok.mean())
    out['mrr_at_10']=float(np.where((r>0)&(r<=10),1/np.maximum(r,1),0).mean())
    for k in (100,300,500,1000,3000):out[f'recall_at_{k}']=float(((r>0)&(r<=k)).mean())
    return out

# 모델별 사용자 query와 전체 상품 표현을 준비하고 seen item을 제외해 점수화한다.
class Scorer:
    def __init__(self,name,model,d,hist,device):
        self.name=name;self.model=model;self.d=d;self.hist=hist;self.device=device
        self.items=None;self.users=None
        if model is not None:
            model.eval()
            with torch.no_grad():
                if name in ('sasrec','esasrec'):
                    # Future-only IDs may have appeared as sampled negatives,
                    # but have no positive train evidence. Keep them neutral in
                    # both sequence queries and target scoring.
                    # 학습 양성이 없는 cold item embedding을 중립값 0으로 맞춘다.
                    model.item.weight[1:][torch.tensor(d.cold,device=device)]=0
                    self.items=model.item.weight[1:].detach().clone()
                elif name=='smore':self.users,self.items=model.forward(model.norm_adj);self.items=self.items.detach().clone()
                else:self.users,self.items=model.representations();self.items=self.items.detach().clone()
                if name!='smore':self.items[torch.tensor(d.cold,device=device)]=0

    def usable(self,u):
        if self.name=='popularity':return False
        if self.name in ('sasrec','esasrec'):return len(self.hist.get(int(u),[]))>0
        return int(u) in self.d.uid_to_train

    @torch.no_grad()
    def scores(self,uids):
        if self.name in ('sasrec','esasrec'):
            seq=torch.tensor(padded(self.hist,uids,self.model.maxlen),device=self.device)
            query=self.model(seq)[:,-1]
        else:
            loc=torch.tensor([self.d.uid_to_train[int(u)] for u in uids],device=self.device)
            query=self.users[loc]
        scores=query@self.items.T
        rows=[];cols=[]
        for j,u in enumerate(uids):
            seq=self.hist.get(int(u),[]);rows.extend([j]*len(seq));cols.extend(seq)
        if rows:scores[torch.tensor(rows,device=self.device),torch.tensor(cols,device=self.device)]=-torch.inf
        return scores

    def popularity_rank(self,u,target):
        hist=self.hist.get(int(u),[])
        assert target not in hist,'target leaked into history'
        return int(self.d.pop_rank[target]-np.sum(self.d.pop_rank[hist]<self.d.pop_rank[target]))

@torch.no_grad()
# 전체 카탈로그에서 held-out target의 결정적 rank와 cohort별 지표를 계산한다.
def evaluate(name,model,d,split,device,batch_size=256,save_to=None,limit=None):
    if split=='test':assert (ART/'selection_lock.json').exists(),'test locked until development is complete'
    targets=pd.read_parquet(DATA/f'{split}_targets.parquet')
    if limit:targets=targets.head(limit).copy()
    hist=d.histories(split);scorer=Scorer(name,model,d,hist,device)
    ranks=np.empty(len(targets),dtype=np.int32)
    active=[]
    for j,row in enumerate(targets.itertuples()):
        if scorer.usable(row.uid):active.append(j)
        else:ranks[j]=scorer.popularity_rank(row.uid,row.iid)
    ids=torch.arange(d.nitems,device=device)
    for start in range(0,len(active),batch_size):
        idx=active[start:start+batch_size];batch=targets.iloc[idx]
        scores=scorer.scores(batch.uid.to_numpy())
        t=torch.tensor(batch.iid.to_numpy(),device=device)
        ts=scores[torch.arange(len(idx),device=device),t]
        assert torch.isfinite(ts).all()
        rr=1+(scores>ts[:,None]).sum(-1)+((scores==ts[:,None])&(ids[None,:]<t[:,None])).sum(-1)
        ranks[idx]=rr.cpu().numpy()
    targets['rank']=ranks;targets['method']=name
    if save_to:
        save_to=Path(save_to);tmp=save_to.with_suffix(save_to.suffix+'.tmp')
        targets.to_parquet(tmp,index=False);os.replace(tmp,save_to)
    out={'all_official_targets':metrics(ranks)}
    for k in (3,5):out[f'train_history_ge_{k}']=metrics(ranks[targets.train_history_length>=k])
    out['personalized_users']=len(active)
    return out,targets

# 전체 catalog에서 뽑되 사용자의 train 양성은 제외해 음성을 만든다.
def negatives(rng,uids,d,nitems):
    neg=rng.integers(0,nitems,size=len(uids))
    for j,u in enumerate(uids):
        while neg[j] in d.hist[int(u)]:neg[j]=rng.integers(nitems)
    return neg

# validation NDCG@10 기준으로 각 backbone의 최선 epoch를 선택한다.
def train(name,lr,d,c,device):
    tag=f'{name}_lr{lr:g}';out=ART/'checkpoints';out.mkdir(parents=True,exist_ok=True)
    bestfile=out/f'{tag}.pt';resume=out/f'{tag}.resume.pt';done=out/f'{tag}.complete.json'
    if done.exists():return load_json(done)
    event(tag,'started')
    model=new_model(name,d,c).to(device);opt=torch.optim.Adam(model.parameters(),lr=lr)
    best=-1.;bad=0;first=1;rng=np.random.default_rng(SEED)
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=False)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);best=state['best'];bad=state['bad'];first=state['epoch']+1
        rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'].cpu())
        if 'cuda_rng' in state:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    seq_model=name in ('sasrec','esasrec')
    seq_u=np.array([u for u,h in d.hist.items() if len(h)>=2])
    size=c['sasrec_batch_size'] if seq_model else c.get(f'{name}_batch_size',512)
    begin=time.monotonic()
    maximum_epochs=c.get(f'{name}_maximum_epochs',c['maximum_epochs'])
    # SASRec 계열은 시퀀스 next-item 손실, 나머지는 interaction pair 손실을 최적화한다.
    for epoch in range(first,maximum_epochs+1):
        model.train();losses=[]
        indices=rng.permutation(len(seq_u) if seq_model else len(d.train))
        for start in range(0,len(indices),size):
            idx=indices[start:start+size];opt.zero_grad(set_to_none=True)
            if seq_model:
                uids=seq_u[idx];seq=np.zeros((len(uids),c['sasrec_maxlen']),dtype=np.int64);pos=seq.copy();neg=seq.copy()
                nneg=64 if name=='esasrec' else 1
                neg=np.zeros((len(uids),c['sasrec_maxlen'],nneg),dtype=np.int64) if nneg>1 else seq.copy()
                for j,u in enumerate(uids):
                    full=d.hist[int(u)];h=full[-c['sasrec_maxlen']-1:];n=len(h)-1
                    seq[j,-n:]=h[:-1]+1;pos[j,-n:]=h[1:]+1
                    v=rng.integers(d.nitems,size=(n,nneg))
                    for k in range(n):
                        for q in range(nneg):
                            while v[k,q] in full:v[k,q]=rng.integers(d.nitems)
                    neg[j,-n:]=v+1 if nneg>1 else v[:,0]+1
                tensors=[torch.tensor(a,device=device) for a in (seq,pos,neg)]
                loss=model.sampled_softmax_loss(*tensors) if name=='esasrec' else model.loss(*tensors)
            else:
                batch=d.train.iloc[idx];n=negatives(rng,batch.uid.to_numpy(),d,d.nitems)
                args=[torch.tensor(x,device=device) for x in (batch.local_uid.to_numpy(),batch.iid.to_numpy(),n)]
                loss=model.calculate_loss(args) if name=='smore' else model.loss(*args,reg=c['regularization'])
            if not torch.isfinite(loss):raise FloatingPointError(f'{tag} nonfinite loss at {epoch}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        event(tag,'epoch',epoch=epoch,loss=float(np.mean(losses)),seconds=round(time.monotonic()-begin,2))
        # 전체 공식 validation target의 NDCG@10으로 저장과 early stopping을 결정한다.
        if epoch%c['validation_every_epochs']==0 or epoch==maximum_epochs:
            result,_=evaluate(name,model,d,'validation',device,c['evaluation_batch_size'])
            value=result['all_official_targets']['ndcg_at_10']
            event(tag,'validation',epoch=epoch,**result['all_official_targets'])
            if value>best:
                best=value;bad=0
                torch.save({'name':name,'state_dict':model.state_dict(),'config':c,'lr':lr,'epoch':epoch,'validation':result,'protocol_sha256':sha(ROOT/'configs/protocol.json')},bestfile)
            else:bad+=1
        state={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'bad':bad,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state()}
        if device=='cuda':state['cuda_rng']=torch.cuda.get_rng_state_all()
        tmp=resume.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,resume)
        if bad>=c['patience']:break
    ck=torch.load(bestfile,map_location='cpu',weights_only=False)
    result={'name':name,'lr':lr,'best_epoch':ck['epoch'],'checkpoint':str(bestfile),'validation':ck['validation'],'elapsed_seconds_this_session':time.monotonic()-begin}
    save_json(done,result);event(tag,'complete',best_epoch=ck['epoch'],validation_ndcg_at_10=best)
    del model,opt
    if device=='cuda':torch.cuda.empty_cache()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['popularity','bpr','lightgcn','sasrec']);args=p.parse_args()
    device=setup();d=RecData();c=load_json(ROOT/'configs/backbones.json')
    # This entrypoint is validation-only. Test is a separate, selection-guarded stage.
    for name in args.models:
        if name=='popularity':
            result,rows=evaluate(name,None,d,'validation',device)
            save_json(ART/'popularity_validation.json',result)
            event('popularity','complete',**result['all_official_targets'])
        else:
            for lr in c['learning_rates']:train(name,lr,d,c,device)

if __name__=='__main__':main()
