"""F5 보강: 생성 품질 — render-FID-lite·다양도·신규성·렌더성공률.

다운스트림 모델링 우위(L1 576 bits)가 생성 품질로도 이어지는가?
각 arm 모델로 200개 생성 → 디코드 → 렌더 → 분포 지표.
l1_bpe는 정확 역매핑 위해 자체 BPE(머지 확장) 사용(SP 유니코드 해킹은 손실).
"""
import sys, io, math
sys.path.insert(0, "/Users/limit/Projects")
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer/.research")

import numpy as np
import torch
import torch.nn.functional as F
import resvg_py
from PIL import Image
import sentencepiece as spm

import f5_data
from f5_run import VanillaLM, train
from gpl_tokenizer.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from gpl_tokenizer.tokenizer.detokenizer import Detokenizer

torch.manual_seed(0); np.random.seed(0)
DEV, RES, NGEN = "cpu", 64, 200
_pt = PrimitiveTokenizer()
_detok = Detokenizer(_pt.vocab, _pt.arcs)
sp_text = spm.SentencePieceProcessor(model_file="/tmp/svg_bpe_5561.model")


# ---- 자체 BPE (정확 역매핑) ----
import collections
def bpe_train(corpus, n):
    seqs=[list(s) for s in corpus]; pairs=collections.Counter()
    for s in seqs:
        for i in range(len(s)-1): pairs[(s[i],s[i+1])]+=1
    merges=[]
    for _ in range(n):
        if not pairs: break
        (a,b),c=max(pairs.items(),key=lambda kv:kv[1])
        if c<2: break
        new=("M",len(merges)); merges.append((a,b,new))
        for s in seqs:
            i=0
            while i<len(s)-1:
                if s[i]==a and s[i+1]==b:
                    if i>0: pairs[(s[i-1],a)]-=1
                    if i+2<len(s): pairs[(b,s[i+2])]-=1
                    s[i]=new; del s[i+1]
                    if i>0: pairs[(s[i-1],new)]+=1
                    if i+1<len(s): pairs[(new,s[i+1])]+=1
                else: i+=1
        pairs.pop((a,b),None)
    return merges
def bpe_apply(seq,merges):
    s=list(seq)
    for a,b,new in merges:
        i=0
        while i<len(s)-1:
            if s[i]==a and s[i+1]==b: s[i:i+2]=[new]
            else: i+=1
    return s
def bpe_expand(sym, table):
    if isinstance(sym,tuple):
        a,b=table[sym]; return bpe_expand(a,table)+bpe_expand(b,table)
    return [sym]


def render(svg):
    png=resvg_py.svg_to_bytes(svg_string=svg,width=RES,height=RES)
    rgba=Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    bg=Image.new("RGBA",rgba.size,(255,255,255,255))
    return np.asarray(Image.alpha_composite(bg,rgba).convert("L"),dtype=np.float64)/255.0


def feats(img):           # 8x8 다운샘플 = 64차원
    im=Image.fromarray((img*255).astype(np.uint8)).resize((8,8))
    return np.asarray(im,dtype=np.float64).flatten()/255.0


def fid_diag(G,R):        # 대각공분산 Fréchet (n=200서 안정)
    mg,mr=G.mean(0),R.mean(0); vg,vr=G.var(0)+1e-6,R.var(0)+1e-6
    return float(((mg-mr)**2).sum()+((np.sqrt(vg)-np.sqrt(vr))**2).sum())


def ink(img): return img<0.9
def iou(a,b):
    u=(a|b).sum(); return (a&b).sum()/u if u else 1.0


@torch.no_grad()
def gen(model,n,max_len=200,temp=0.9,topk=40):
    model.eval(); outs=[]
    for _ in range(n):
        x=torch.tensor([[1]],device=DEV)
        for _ in range(max_len):
            lg=model(x)[:,-1,:]/temp
            v,_=torch.topk(lg,topk); lg[lg<v[:,-1:]]=-1e9
            p=F.softmax(lg,-1); nx=torch.multinomial(p,1)
            x=torch.cat([x,nx],1)
            if nx.item()==2: break
        outs.append(x[0,1:].tolist())
    return outs


def decode(arm,toks,table=None):
    toks=[t for t in toks if t!=2]
    if arm=="char":
        return sp_text.decode([t for t in toks if isinstance(t,int) and t<5561])
    if arm=="l1":
        ids=[t for t in toks if isinstance(t,int)]
        return _detok.to_svg_document(ids)
    if arm=="l1_bpe":
        ids=[]
        for t in toks: ids+=bpe_expand(t,table)
        return _detok.to_svg_document([i for i in ids if isinstance(i,int)])


def main():
    (trd,ted)=f5_data.load(n_train=1500,n_test=300,max_len=160)
    # 실데이터 렌더 피처(FID 기준) + 학습셋 잉크마스크(신규성)
    real_f=[]; train_ink=[]
    for svg in ted["svg"]:
        try: real_f.append(feats(render(svg)))
        except: pass
    for svg in trd["svg"][:400]:
        try: train_ink.append(ink(render(svg)))
        except: pass
    real_f=np.array(real_f)

    merges=bpe_train(trd["l1"],1000); table={n:(a,b) for a,b,n in merges}
    arms={
        "char":   (trd["char_bpe"],5561,None),
        "l1":     (trd["l1"],5561,None),
        "l1_bpe": ([bpe_apply(s,merges) for s in trd["l1"]],5561+len(merges),table),
    }
    # l1_bpe 심볼→연속 id 매핑(임베딩용)
    sym2id={}
    def remap(seq):
        out=[]
        for s in seq:
            if isinstance(s,int): out.append(s)
            else:
                if s not in sym2id: sym2id[s]=5561+len(sym2id)
                out.append(sym2id[s])
        return out
    id2sym={}
    if True:
        arms["l1_bpe"]=([remap(s) for s in arms["l1_bpe"][0]],5561+1000,table)
        id2sym={v:k for k,v in sym2id.items()}

    print(f"{'arm':<8}{'render%':>9}{'FID↓':>9}{'diversity↑':>12}{'novelty↑':>10}")
    for arm,(seqs,V,tab) in arms.items():
        torch.manual_seed(0); np.random.seed(0)
        model=VanillaLM(V).to(DEV); train(model,seqs,epochs=25,max_len=200)
        gens=gen(model,NGEN)
        gf=[]; masks=[]; ok=0
        for g in gens:
            if arm=="l1_bpe":
                g=[id2sym.get(t,t) for t in g]   # 연속 id → 심볼 복원
            try:
                svg=decode(arm,g,tab)
                if not svg or "<" not in svg: continue
                im=render(svg)
                if ink(im).sum()<8: continue   # 백지 제외
                ok+=1; gf.append(feats(im)); masks.append(ink(im))
            except: continue
        gf=np.array(gf)
        fid=fid_diag(gf,real_f) if len(gf)>5 else float("nan")
        # 다양도: 표본 쌍 1-IoU
        div=float(np.mean([1-iou(masks[i],masks[j]) for i in range(len(masks))
                           for j in range(i+1,min(i+6,len(masks)))])) if len(masks)>1 else 0
        # 신규성: 학습셋 최근접 1-maxIoU
        nov=float(np.mean([1-max(iou(m,t) for t in train_ink[:200]) for m in masks[:80]])) if masks else 0
        print(f"{arm:<8}{ok/NGEN*100:>8.0f}%{fid:>9.3f}{div:>12.3f}{nov:>10.3f}",flush=True)

    print("\n해석: FID↓(실데이터 분포 근접), 다양도↑(모드붕괴 아님), 신규성↑(암기 아님).")
    print(" L1이 FID 최저+다양도 유지면 '모델링 우위(576 bits)가 생성 품질로 이어짐'.")


if __name__=="__main__":
    main()
