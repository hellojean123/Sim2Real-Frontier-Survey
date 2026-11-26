#!/usr/bin/env python3
"""Supplement the trajectory-diversity survey with source-PDF figure screenshots.
Only embed images actually captured. Preserve old assets and prose.
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib, io, json, re, subprocess, unicodedata
import fitz, requests
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/"docs/轨迹多样性_生成机制与质量评估专题综述_2026-11-26.md"
DEST=ROOT/"assets/paper-screenshots/trajectory-diversity"
DEST.mkdir(parents=True,exist_ok=True)
MAN=DEST/"manifest.json"
EXTRA={
"MimicGen":["https://arxiv.org/pdf/2310.17596"],
"SkillMimicGen / SkillGen":["https://arxiv.org/pdf/2410.18907"],
"IntervenGen":["https://arxiv.org/pdf/2405.01472","https://autolab.berkeley.edu/assets/publications/media/IROS24_2512_FI.pdf"],
"D3IL":["https://arxiv.org/pdf/2402.14606"],
"What Matters":["https://arxiv.org/pdf/2506.13536"],
}
OLD={"RoboTwin 2.0":"RoboTwin","SCIZOR":"SCIZOR","TUCO":"TUCO"}
CAP=re.compile(r"^\s*(?:fig(?:ure)?\.?)\s*(\d+)\s*[:.\-–—]?\s*(.*)",re.I)
METHOD=re.compile(r"overview|framework|pipeline|architecture|method|algorithm|system|illustration|generation|workflow|approach",re.I)
def pdf_urls(name,url):
    urls=EXTRA.get(name,[]).copy()
    m=re.search(r"arxiv\.org/(?:abs|html|pdf)/(\d+\.\d+(?:v\d+)?)",url)
    if m:
        ident=m.group(1)
        urls.extend(["https://arxiv.org/pdf/"+ident,"https://export.arxiv.org/pdf/"+ident])
    m=re.search(r"proceedings\.mlr\.press/(v\d+)/([^/]+)\.html",url)
    if m:
        v,p=m.groups()
        urls.extend([f"https://raw.githubusercontent.com/mlresearch/{v}/main/assets/{p}/{p}.pdf",f"https://proceedings.mlr.press/{v}/{p}/{p}.pdf"])
    if "-Abstract-Conference.html" in url: urls.append(url.replace("-Abstract-Conference.html","-Paper-Conference.pdf").replace("/hash/","/pdf/"))
    if "openaccess.thecvf.com" in url: urls.append(url.replace("/html/","/papers/").replace("_paper.html","_paper.pdf"))
    return list(dict.fromkeys(urls))
def words(s):
    return set(re.findall(r"[a-z0-9]{3,}",unicodedata.normalize("NFKC",s).lower()))
def load(name,title,url):
    errs=[]
    for u in pdf_urls(name,url):
        try:
            r=requests.get(u,timeout=(10,35),headers={"User-Agent":"Mozilla/5.0 Academic figure capture"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"): raise ValueError("not a PDF")
            d=fitz.open(stream=r.content,filetype="pdf")
            text=d[0].get_text()
            target=words(title)
            overlap=len(target & words(text))/max(len(target),1)
            if overlap<.65: raise ValueError(f"title validation failed ({overlap:.2f})")
            return d,u,hashlib.sha256(r.content).hexdigest()
        except Exception as e: errs.append(f"{u}: {e}")
    raise ValueError("; ".join(errs) or "No verified PDF URL")
def captions(d):
    hits=[]
    for i,p in enumerate(d):
        if i>24: break
        for b in p.get_text("dict")["blocks"]:
            if "lines" not in b: continue
            lines=b["lines"]
            for j,l in enumerate(lines):
                t="".join(s["text"] for s in l["spans"]).strip()
                m=CAP.match(t)
                if not m: continue
                # Distinguish caption labels from inline references at the start of a prose line.
                if not re.match(r"^\s*(?:fig(?:ure)?\.?)\s*\d+\s*[:.\-–—]",t,re.I):
                    if re.match(r"^(shows?|illustrates?|presents?|depicts?|reports?|demonstrates?|is|provides?|compares?)\b",m.group(2),re.I): continue
                rect=fitz.Rect(l["bbox"])
                caption=t
                for nxt in lines[j+1:j+4]:
                    nt="".join(s["text"] for s in nxt["spans"]).strip()
                    if CAP.match(nt): break
                    if nxt["bbox"][1]-rect.y1>8: break
                    caption+=" "+nt
                    rect|=fitz.Rect(nxt["bbox"])
                hits.append(dict(number=int(m.group(1)),page=i,caption=caption[:400],box=list(rect)))
    # Prefer actual caption lines with punctuation; keep first true label per number.
    out={}
    for h in hits: out.setdefault(h["number"],h)
    return list(out.values())
def render(d,h,path):
    p=d[h["page"]]; box=fitz.Rect(h["box"]); W,H=p.rect.width,p.rect.height
    # Prefer a conservative crop when a substantial image directly abuts the caption.
    # Otherwise retain the full original figure-bearing page: avoids clipped vector diagrams.
    candidates=[]
    for im in p.get_image_info():
        r=fitz.Rect(im["bbox"])
        if r.width<60 or r.height<32: continue
        if -4<=box.y0-r.y1<=45 and r.y0<box.y0 and r.x0<box.x1 and r.x1>box.x0:
            candidates.append(r)
    # Figure-anchored extraction: a paper page is NOT a figure screenshot.
    # Detect one/two-column caption geometry and take the closest artwork above.
    # Preserve extra padding for vector diagrams whose boxes cannot be inferred.
    if box.x0 >= W*.49:
        x0,x1=W*.50,W-14
        band=255
        mode="right_column_anchored_crop"
    elif box.x1 <= W*.51:
        x0,x1=14,W*.50
        band=255
        mode="left_column_anchored_crop"
    else:
        x0,x1=20,W-20
        band=370
        mode="full_width_anchored_crop"
    top=max(20,box.y0-band)
    bottom=min(H-10,box.y1+13)
    crop=fitz.Rect(x0,top,x1,bottom)
    if candidates:
        combined=fitz.Rect(candidates[0])
        for r in candidates[1:]: combined|=r
        if len(candidates)==1 and combined.width>=W*.30 and combined.height>=65:
            # Keep entire raster panel plus its caption; do not lose peripheral labels.
            crop=fitz.Rect(max(12,min(combined.x0,box.x0,crop.x0)-7),
                           max(15,min(combined.y0,top)-7),
                           min(W-10,max(combined.x1,box.x1,crop.x1)+7),
                           min(H-8,box.y1+13))
            mode="caption_adjacent_raster_crop"
    if crop.height<65: raise ValueError("Figure crop too short")
    pix=p.get_pixmap(matrix=fitz.Matrix(2.5,2.5),clip=crop,alpha=False)
    im=Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    im.thumbnail((2400,2800),Image.Resampling.LANCZOS)
    im.save(path,"JPEG",quality=94,optimize=True)
    with Image.open(path) as check: check.verify()
    return {"path":str(path.relative_to(ROOT)),"figure":h["number"],"pdf_page":h["page"]+1,"caption":h["caption"],"crop_pdf_points":list(crop),"capture_mode":mode,"bytes":path.stat().st_size}
def capture(item):
    name,title,url=item
    slug=re.sub(r"[^a-zA-Z0-9]+","-",name).strip("-")
    try:
        d,source,sha=load(name,title,url)
        hits=captions(d)
        if not hits: raise ValueError("No figure caption confidently detected")
        first=next((x for x in hits if x["number"]==1),hits[0])
        rest=[x for x in hits if x["number"]!=first["number"]]
        # Prefer a method diagram; otherwise use the second available original figure.
        second=next((x for x in rest if METHOD.search(x["caption"]) and x["page"]<=first["page"]+5),None)
        if second is None and rest: second=rest[0]
        out={"pdf":source,"pdf_sha256":sha,"screenshots":{}}
        for role,h in [("overview",first),("architecture",second)]:
            if h is not None:
                out["screenshots"][role]=render(d,h,DEST/f"{slug}-{role}.jpg")
        if second is None: out["warning"]="Only one confidently located original figure; no duplicate or invented second figure."
        print(name,[(k,v["figure"],v["pdf_page"]) for k,v in out["screenshots"].items()],flush=True)
        return name,out
    except Exception as e:
        print("FAILED",name,str(e)[:240],flush=True)
        return name,{"error":str(e)[:1200],"screenshots":{}}
def main():
    md=DOC.read_text(encoding="utf-8")
    blocks=re.split(r"(?m)(^### \d+\.\d+ .+$)",md)
    items=[]
    for h,b in zip(blocks[1::2],blocks[2::2]):
        if "#### ① 论文基本信息" not in b: continue
        name=re.sub(r"^### \d+\.\d+ ","",h)
        if name in OLD: continue
        title=re.search(r"\*\*原题\*\*：(.+)",b).group(1)
        url=re.search(r"\[论文\]\(([^)]+)\)",b).group(1)
        items.append((name,title,url))
    old=json.loads(MAN.read_text()) if MAN.exists() else {"papers":{}}
    results={}
    todo=[]
    for item in items:
        entry=old["papers"].get(item[0],{})
        ss=entry.get("screenshots",{})
        if old.get("crop_algorithm_version")==2 and len(ss)==2 and all((ROOT/s["path"]).is_file() for s in ss.values()): results[item[0]]=entry
        else: todo.append(item)
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        results.update(dict(pool.map(capture,todo)))
    count=0; complete=0
    for i in range(1,len(blocks),2):
        name=re.sub(r"^### \d+\.\d+ ","",blocks[i])
        b=blocks[i+1]
        if "#### ① 论文基本信息" not in b or name in OLD: continue
        entry=results[name]; ss=entry.get("screenshots",{})
        count+=len(ss); complete+=len(ss)==2
        if not ss: continue
        lines=[]
        for role,s in ss.items():
            label="主图" if role=="overview" else "方法/第二关键图"
            lines.extend([f"**{label}｜Figure {s['figure']}**",
                f"![{name} Figure {s['figure']}](../{s['path']})",
                f"*原论文 Figure {s['figure']}，PDF 第 {s['pdf_page']} 页；{'整页保真截图' if s['capture_mode']=='full_figure_page' else '原图及图注裁切'}。[原文对应页]({entry['pdf']}#page={s['pdf_page']})。*"])
        if len(ss)<2: lines.append("**图源说明：** 仅定位到一张原论文图，未用重复图或生成图填充。")
        lines.append("[截图来源清单](../assets/paper-screenshots/trajectory-diversity/manifest.json)。自动定位图号，裁切尚需人工逐图复核；第二图不一定是网络架构。")
        replacement="#### ④ 方法总览与原论文图\n\n"+"\n\n".join(lines)+"\n\n"
        blocks[i+1]=re.sub(r"#### ④ 方法总览与原论文图\n[\s\S]*?(?=#### ⑤)",lambda _:replacement,b,count=1)
    md="".join(blocks)
    # Keep all substantive analysis intact and correct only screenshot-status statements.
    status=f"截图补充：本专题 34 篇核心论文，复用 3 篇的 6 张原图；新增已保存 {count} 张原论文截图（{complete}/{len(items)} 篇新增论文具备双图）。图源清单记录失败、图号、PDF 页码及截取坐标；自动裁切未逐图人工复核。"
    md=md.replace("对旧文沿用仓库已有图，对新文提供原文/项目图入口及中文流程，不声称新截图已保存。",status)
    md=md.replace("新论文原始图片通过原文/项目入口阅读，未下载截图到仓库。",status)
    md=re.sub(r"截图补充：本专题 34 篇核心论文，复用 3 篇的 6 张原图；新增已保存 \d+ 张原论文截图（\d+/\d+ 篇新增论文具备双图）。图源清单记录失败、图号、PDF 页码及截取坐标；自动裁切未逐图人工复核。",lambda _:status,md)
    MAN.write_text(json.dumps({"source":"original PDFs only","crop_algorithm_version":2,"quality_note":"Caption-anchored crops, automated; still require per-paper visual review","new_papers":len(items),"new_images":count,"complete_pairs":complete,"papers":results},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    DOC.write_text(md,encoding="utf-8")
    # The existing figure workflow stages assets itself; stage the matching Markdown here
    # so figure files, provenance and links are published in the same commit.
    subprocess.run(["git","add","--",str(DOC.relative_to(ROOT))],cwd=ROOT,check=True)
    print(f"NEW {count}/{2*len(items)} images; {complete}/{len(items)} pairs; existing 6 reused",flush=True)
if __name__=="__main__": main()
