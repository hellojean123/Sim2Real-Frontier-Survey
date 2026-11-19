#!/usr/bin/env python3
"""Capture attributed figure screenshots from the original paper PDFs.

Outputs images + provenance manifest. Do not relabel these as hand-drawn figures.
This script runs only when explicitly triggered by the paired GitHub workflow.
"""
from __future__ import annotations
import json, re, time
from pathlib import Path
from urllib.parse import urlparse
import fitz
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "paper-screenshots"
DEST.mkdir(parents=True, exist_ok=True)

PAPERS = [
    ("GSWorld", ["https://3dgsworld.github.io/resources/gsworld.pdf",
                "https://arxiv.org/pdf/2510.20813"], 1, 2),
    ("RE3SIM", ["https://arxiv.org/pdf/2502.08645"], 1, 2),
    ("RoboRender", ["https://arxiv.org/pdf/2610.09254"], 1, 2),
    ("TabletopGen", ["https://arxiv.org/pdf/2512.01204v4", "https://arxiv.org/pdf/2512.01204"], 1, 2),
    ("EmbodiedGenV2", ["https://arxiv.org/pdf/2607.07459v3", "https://arxiv.org/pdf/2607.07459"], 1, 2),
    ("SimFoundry", ["https://arxiv.org/pdf/2606.28276v4", "https://arxiv.org/pdf/2606.28276"], 1, 2),
    ("SPARR", ["https://arxiv.org/pdf/2602.23253"], 1, 3),
    ("SoMA", ["https://raw.githubusercontent.com/mlresearch/v306/main/assets/huang26bh/huang26bh.pdf"], 1, 2),
    ("PIN-WM", ["https://www.roboticsproceedings.org/rss21/p153.pdf"], 1, 2),
    ("RoboTwin", ["https://raw.githubusercontent.com/mlresearch/v306/main/assets/chen26j/chen26j.pdf",
                 "https://robotwin-platform.github.io/paper.pdf"], 1, 2),
    ("RoboDream", ["https://arxiv.org/pdf/2606.02577"], 1, 2),
    ("R2R2R", ["https://raw.githubusercontent.com/mlresearch/v305/main/assets/yu25a/yu25a.pdf"], 1, 2),
    ("SCIZOR", ["https://arxiv.org/pdf/2505.22626"], 1, 2),
    ("MetaGenSim", ["https://arxiv.org/pdf/2610.19479"], 1, 2),
    ("CoTraining", ["https://www.roboticsproceedings.org/rss21/p109.pdf"], 1, 2),
]
# Manual, PDF-point crop rectangles for papers with image-only captions and
# layouts where Figure 1 spans both columns despite a narrow caption line.
MANUAL_CROPS = {
    # These page/figure rectangles were verified against the three PDFs
    # supplied on 2026-11-19. Coordinates are in PDF points (zero-based page).
    "TabletopGen": {
        "overview": (1, [125, 108, 490, 375], 1, "Fig. 1: Tabletop interactive scene, scene expansion and data synthesis"),
        "architecture": (5, [130, 105, 495, 414], 2, "Fig. 2: Generative instance extraction, DRO/TSA alignment and interactive simulation"),
    },
    "EmbodiedGenV2": {
        "overview": (0, [70, 305, 548, 736], 1, "Figure 1: Task-driven, large-scale and Vibe Coding 3D worlds"),
        "architecture": (3, [70, 67, 539, 502], 2, "Figure 2: Sim-ready asset-generation and cross-simulator export"),
    },
    "SimFoundry": {
        "overview": (0, [65, 225, 532, 506], 1, "Figure 1: Real-to-sim digital twin and object/scene/task cousins"),
        "architecture": (3, [58, 80, 541, 466], 2, "Figure 2: Extraction, scene generation and cousin augmentations"),
    },
    "PIN-WM": {
        "overview": (0, [305, 195, 575, 403], 1, "Fig. 1: PIN-WM learning and physical digital cousins"),
        "architecture": (3, [42, 54, 570, 353], 2, "Fig. 2: Physics-informed Real2Sim2Real framework"),
    },
    "MetaGenSim": {
        "overview": (0, [32, 341, 585, 615], 1, "Figure 1: Active Generative Simulation overview"),
    },
    "SPARR": {
        "architecture": (3, [48, 51, 308, 228], 3, "Fig. 3: Asymmetric base and residual policy combination"),
    },
    "SCIZOR": {
        "architecture": (3, [52, 64, 561, 303], 2, "Figure 2: SCIZOR architecture for suboptimal-transition removal and deduplication"),
    },
    "CoTraining": {
        "overview": (0, [305, 190, 577, 590], 1, "Fig. 1: Sim-and-Real Co-Training overview"),
        "architecture": (3, [44, 47, 574, 280], 2, "Fig. 2: Method overview: task-aware digital cousins and sim-real policy co-training"),
    },
}
REFRESH = set()  # Keep prior verified screenshots unchanged when adding papers

FIG_RE = re.compile(r"^\s*(?:fig\.?|figure)\s*(\d+)\s*[:.\-\u2014]", re.IGNORECASE)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; AcademicFigureCapture/1.0; +https://github.com/hellojean123/Sim2Real-Frontier-Survey)"})

def download_pdf(urls):
    errors = []
    for u in urls:
        candidate_urls = [u]
        match = re.search(r"arxiv\.org/pdf/(\d+\.\d+)", u)
        if match:
            candidate_urls.extend([
                "https://export.arxiv.org/pdf/" + match.group(1),
                "https://arxiv.org/pdf/" + match.group(1) + ".pdf",
            ])
        for candidate in candidate_urls:
            try:
                r = SESSION.get(candidate, timeout=(20, 100))
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    raise ValueError("response is not PDF: " + r.headers.get("Content-Type", "unknown"))
                if len(r.content) > 80_000_000:
                    raise ValueError("PDF too large (>80 MB)")
                return r.content, candidate
            except Exception as exc:
                errors.append(f"{candidate}: {str(exc)[:130]}")
        time.sleep(0.3)
    raise RuntimeError("; ".join(errors))

def scan_captions(doc):
    hits = []
    for page_no in range(min(len(doc), 15)):
        page = doc[page_no]
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = "".join(span.get("text", "") for span in line["spans"]).strip()
                m = FIG_RE.match(line_text)
                if m:
                    rect = fitz.Rect(line["bbox"])
                    hits.append({"figure": int(m.group(1)), "page": page_no,
                                 "caption": line_text[:220], "box": [rect.x0, rect.y0, rect.x1, rect.y1]})
    return hits

def screenshot_figure(doc, cap, out_file):
    page = doc[cap["page"]]
    page_w, page_h = page.rect.width, page.rect.height
    cx0, cy0, cx1, cy1 = cap["box"]
    # Choose left/right column only when caption definitely begins in that half.
    if cx0 > page_w * 0.52:
        x0, x1 = page_w * .51, page_w - 18
    elif cx1 < page_w * 0.48:
        x0, x1 = 18, page_w * .49
    else:
        x0, x1 = 25, page_w - 25
    tall = 270 if (x1-x0) > page_w*0.7 else 220
    top = max(32, cy0 - tall)
    if cap["page"] == 0:
        top = max(top, 105)  # remove paper header and authors from page-1 screenshot
    # If the PDF has a substantial figure raster just above the caption, preserve its top edge.
    boxes = []
    for img in page.get_images(full=True):
        try:
            for rect in page.get_image_rects(img[0]):
                if rect.width < 45 or rect.height < 28:
                    continue
                if rect.y1 < cy0 + 6 and cy0-rect.y1 < 55 and rect.y0 > cy0-460:
                    if rect.x1 > x0 and rect.x0 < x1:
                        boxes.append(rect)
        except Exception:
            pass
    if boxes:
        imtop = min(r.y0 for r in boxes)
        top = max(30, min(top, imtop - 8))
    bottom = min(page_h-20, cy1+10)
    if bottom-top < 60:
        raise ValueError("invalid figure rectangle")
    rect = fitz.Rect(x0, top, x1, bottom)
    pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), clip=rect, alpha=False)
    import io
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    # Stay readable in GitHub preview but avoid unwieldy 4K JPEGs.
    if img.width > 1900:
        img.thumbnail((1900, 1900), Image.Resampling.LANCZOS)
    img.save(out_file, "JPEG", quality=91, optimize=True, progressive=True)
    return {"page": cap["page"]+1, "figure": cap["figure"],
            "caption": cap["caption"], "crop_pdf_points": [round(v,1) for v in rect]}

def manual_screenshot(doc, spec, output):
    page_index, coords, figure_no, caption = spec
    page = doc[page_index]
    r = fitz.Rect(coords) & page.rect
    scale = 3.0 if r.width < 350 else 2.2
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=r, alpha=False)
    import io
    im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    if im.width > 1900:
        im.thumbnail((1900,1900), Image.Resampling.LANCZOS)
    im.save(output, "JPEG", quality=91, optimize=True, progressive=True)
    return {"page": page_index+1, "figure": figure_no,
            "caption": caption, "crop_pdf_points": [round(v,1) for v in r]}

def main():
    old_path=DEST/"manifest.json"
    previous=json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {"papers":{}}
    manifest = {"generated_from": "original paper PDFs, cropped as screenshots with PyMuPDF",
                "note": "Figure number and PDF page are reported individually; images may include immediate neighboring captions.",
                "papers": {}, "failures": []}
    for slug, urls, overview_no, arch_no in PAPERS:
        print(f"Processing {slug}", flush=True)
        if slug not in REFRESH and slug in previous.get("papers", {}):
            old_entry=previous["papers"][slug]
            if all((ROOT/screenshot["path"]).exists() for screenshot in old_entry.get("screenshots",{}).values()) and len(old_entry.get("screenshots",{})) == 2:
                manifest["papers"][slug]=old_entry
                print("  reusing verified previous screenshot pair",flush=True)
                continue
        try:
            raw, paperurl = download_pdf(urls)
            doc = fitz.open(stream=raw, filetype="pdf")
            found = scan_captions(doc)
            group = {"pdf": paperurl, "available_figure_numbers": sorted(set(c["figure"] for c in found)),
                     "screenshots": {}}
            used = set()
            for kind, target in (("overview", overview_no), ("architecture", arch_no)):
                candidates = [c for c in found if c["figure"] == target and (c["page"],c["figure"]) not in used]
                if not candidates and kind == "architecture":
                    candidates = [c for c in found if c["figure"] not in (overview_no,) and
                                  (c["page"],c["figure"]) not in used]
                manual = MANUAL_CROPS.get(slug,{}).get(kind)
                if not candidates and manual:
                    candidates = [{"figure":manual[2],"page":manual[0],"caption":manual[3],"box":manual[1]}]
                if not candidates:
                    manifest["failures"].append({"paper": slug, "kind": kind, "error": "caption not located"})
                    continue
                cap = candidates[0]
                filename = f"{slug}-{kind}.jpg"
                manual = MANUAL_CROPS.get(slug,{}).get(kind)
                details = manual_screenshot(doc,manual,DEST/filename) if manual else screenshot_figure(doc,cap,DEST/filename)
                used.add((cap["page"],cap["figure"]))
                group["screenshots"][kind] = {"path": "assets/paper-screenshots/"+filename, **details}
                print(f"  {kind}: Fig. {cap['figure']} page {cap['page']+1}",flush=True)
            manifest["papers"][slug] = group
        except Exception as e:
            manifest["failures"].append({"paper": slug, "error": str(e)[:450]})
            print(f"  FAILURE {str(e)[:250]}", flush=True)
    (DEST/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    total=sum(len(x["screenshots"]) for x in manifest["papers"].values())
    print(f"FINISHED: {total}/30 original screenshots, {len(manifest['failures'])} warnings", flush=True)
    if total < 30:
        raise SystemExit("Fewer than 30 actual screenshots extracted. Manual review needed.")
if __name__=="__main__":
    main()
