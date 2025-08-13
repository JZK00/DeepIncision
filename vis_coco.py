#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import argparse
from typing import Dict, List, Any, Optional, Tuple

# ============ 默认类别配置（按你的7类） ============
CLASS_NAMES = [
    "No abnormal",
    "Redness",
    "Suppuration",
    "Scab",
    "Tension blisters",
    "Dehiscenced",
    "Ecchymosis around the incision",
]
ID_OFFSET = 1  # 如果你的 category_id 从 0 开始，请改为 0

USE_ID2NAME_MAP = False
ID2NAME_MAP = {
    1: "No abnormal",
    2: "Redness",
    3: "Suppuration",
    4: "Scab",
    5: "Tension blisters",
    6: "Dehiscenced",
    7: "Ecchymosis around the incision",
}

# 固定颜色（BGR）
PRED_COLOR = (0, 0, 139)      # 深红
GT_COLOR   = (230, 216, 173)  # 浅蓝（RGB 173,216,230 的 BGR）

# ===== 可选：手动覆盖自适应（设为 >0 则固定，不再自适应）=====
MANUAL_LINE_THICKNESS = 0     # 例如设为 6 可固定线宽；0 表示自动
MANUAL_FONT_SCALE     = 0.0   # 例如设为 1.2 可固定字体大小；0 表示自动
MANUAL_FONT_THICKNESS = 0     # 例如设为 2 可固定粗细；0 表示自动

# ============ 工具函数 ============


def build_img_lookup_from_gt(gt_json_path: str):
    from pycocotools.coco import COCO
    coco = COCO(gt_json_path)
    ids = coco.getImgIds()
    imgs = coco.loadImgs(ids)
    id2name, name2id, stem2id = {}, {}, {}
    for im in imgs:
        iid = int(im["id"])
        fn = im.get("file_name", "")
        id2name[iid] = fn
        if fn:
            name2id[fn] = iid
            stem, _ = os.path.splitext(os.path.basename(fn))
            if stem:
                stem2id[stem] = iid
    return coco, id2name, name2id, stem2id

def resolve_image_path(image_id_any, id2name: Optional[Dict[int, str]], img_dir: str) -> Optional[str]:
    common_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    if isinstance(image_id_any, int):
        if not id2name:
            return None
        fname = id2name.get(image_id_any)
        if not fname:
            return None
        p = os.path.join(img_dir, fname)
        return p if os.path.exists(p) else None
    s = str(image_id_any)
    root, ext = os.path.splitext(s)
    if ext.lower() in common_exts:
        p = os.path.join(img_dir, s)
        if os.path.exists(p):
            return p
    for e in common_exts:
        p = os.path.join(img_dir, s + e)
        if os.path.exists(p):
            return p
    try:
        for fn in os.listdir(img_dir):
            stem, _ = os.path.splitext(fn)
            if stem == s:
                p = os.path.join(img_dir, fn)
                if os.path.exists(p):
                    return p
    except FileNotFoundError:
        pass
    return None

def class_name_from_id(category_id: int) -> str:
    if USE_ID2NAME_MAP:
        return ID2NAME_MAP.get(int(category_id), f"class_{category_id}")
    idx = int(category_id) - ID_OFFSET
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return f"class_{category_id}"

# ===== 自适应可视化参数 =====
def get_vis_params(img_shape,
                   base_short_side: int = 800,
                   base_line: int = 3,
                   base_font_scale: float = 0.8,
                   line_minmax: Tuple[int, int] = (2, 20),
                   font_scale_minmax: Tuple[float, float] = (0.6, 4.0)) -> Tuple[int, float, int, int]:
    """
    根据图像分辨率自适应返回：
      line_thickness, font_scale, font_thickness, pad
    以短边=800px 时 baseline: line=3, font_scale=0.8。
    结果做了上下限裁剪，并给出文字背景 padding。
    """
    h, w = img_shape[:2]
    short_side = max(1, min(h, w))
    scale = short_side / float(base_short_side)

    if MANUAL_LINE_THICKNESS > 0:
        line_thickness = MANUAL_LINE_THICKNESS
    else:
        line_thickness = int(round(base_line * scale))
        line_thickness = max(line_minmax[0], min(line_thickness, line_minmax[1]))

    if MANUAL_FONT_SCALE > 0:
        font_scale = MANUAL_FONT_SCALE
    else:
        font_scale = base_font_scale * scale
        font_scale = max(font_scale_minmax[0], min(font_scale, font_scale_minmax[1]))

    if MANUAL_FONT_THICKNESS > 0:
        font_thickness = MANUAL_FONT_THICKNESS
    else:
        font_thickness = max(1, int(round(line_thickness * 0.6)))

    pad = max(2, int(round(line_thickness * 0.8)))
    return line_thickness, font_scale, font_thickness, pad

def draw_box_with_label(img, bbox, text: str, color: Tuple[int, int, int],
                        line_thickness: int, font_scale: float, font_thickness: int, pad: int):
    x, y, w, h = map(float, bbox)  # COCO: [x, y, w, h]
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    y_text = max(th + 2 * pad, y1)  # 防止超出顶边
    # 文本背景
    cv2.rectangle(img,
                  (x1, y_text - th - 2 * pad),
                  (x1 + tw + 2 * pad, y_text),
                  color, -1)
    cv2.putText(img, text,
                (x1 + pad, y_text - pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), font_thickness, cv2.LINE_AA)

def draw_pred(img, bbox, label: str, score: float,
              line_thickness: int, font_scale: float, font_thickness: int, pad: int):
    txt = f"{label} {score:.2f}"
    draw_box_with_label(img, bbox, txt, PRED_COLOR,
                        line_thickness, font_scale, font_thickness, pad)

def draw_gt(img, bbox,
            line_thickness: int, font_scale: float, font_thickness: int, pad: int):
    txt = "ground truth"
    draw_box_with_label(img, bbox, txt, GT_COLOR,
                        line_thickness, font_scale, font_thickness, pad)

# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="Visualize COCO-style predictions (bbox) with resolution-adaptive styling.")
    parser.add_argument("--pred", type=str, required=True, help="Prediction JSON path.")
    parser.add_argument("--imgdir", type=str, required=True, help="Directory of images.")
    parser.add_argument("--outdir", type=str, default="vis_preds", help="Output directory.")
    parser.add_argument("--gt", type=str, default=None, help="Optional COCO GT json to draw ground truth and map image_id.")
    parser.add_argument("--score_thr", type=float, default=0.0, help="Score threshold.")
    parser.add_argument("--only_classes", type=str, default="", help="Comma-separated class names to keep.")
    parser.add_argument("--topk", type=int, default=0, help="Keep top-K predictions per image (0 means all).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.pred, "r") as f:
        preds = json.load(f)
    if isinstance(preds, dict) and "annotations" in preds:
        preds = preds["annotations"]

    coco_gt = None
    id2name = None
    if args.gt:
        try:
            coco_gt, id2name, _, _ = build_img_lookup_from_gt(args.gt)
        except Exception as e:
            print(f"[Warn] Fail to read GT json: {e}. Will resolve images by filename or stem only.")

    imgid2preds: Dict[Any, List[Dict[str, Any]]] = {}
    for p in preds:
        imgid2preds.setdefault(p["image_id"], []).append(p)

    keep_set = None
    if args.only_classes.strip():
        keep = [s.strip() for s in args.only_classes.split(",") if s.strip()]
        keep_set = set(keep)

    total, ok = 0, 0

    for image_id_any, plist in imgid2preds.items():
        total += 1
        img_path = resolve_image_path(image_id_any, id2name, args.imgdir)
        if img_path is None:
            print(f"[Skip] Cannot locate image for image_id={image_id_any}")
            continue

        out_path = os.path.join(args.outdir, os.path.basename(img_path))
        if (not args.overwrite) and os.path.exists(out_path):
            ok += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[Skip] Cannot read image: {img_path}")
            continue

        # —— 自适应参数（每张图单独计算）——
        line_thk, f_scale, f_thk, pad = get_vis_params(img.shape)

        # 先画 GT
        if coco_gt is not None and id2name is not None:
            img_id_int = None
            if isinstance(image_id_any, int):
                img_id_int = image_id_any
            else:
                file_name = os.path.basename(img_path)
                for k, v in id2name.items():
                    if v == file_name:
                        img_id_int = k
                        break
            if img_id_int is not None:
                ann_ids = coco_gt.getAnnIds(imgIds=[img_id_int])
                anns = coco_gt.loadAnns(ann_ids)
                for ann in anns:
                    if "bbox" in ann:
                        draw_gt(img, ann["bbox"], line_thk, f_scale, f_thk, pad)

        # 再画预测
        cur = []
        for p in plist:
            score = float(p.get("score", 1.0))
            if score < args.score_thr:
                continue
            cname = class_name_from_id(int(p["category_id"]))
            if keep_set is not None and cname not in keep_set:
                continue
            cur.append((p, cname, score))
        if args.topk > 0 and len(cur) > args.topk:
            cur = sorted(cur, key=lambda t: t[2], reverse=True)[:args.topk]

        for p, cname, score in cur:
            draw_pred(img, p["bbox"], cname, score, line_thk, f_scale, f_thk, pad)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)
        ok += 1

    print(f"Done: grouped {total} image_id, saved {ok} images to {args.outdir}")

if __name__ == "__main__":
    import cv2
    main()
