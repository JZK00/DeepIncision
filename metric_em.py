import os
import pickle
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import csv
import os
import json
from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion


def compute_iou_np(boxes1, boxes2):
    """boxes: [N, 4] format: x1, y1, x2, y2"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / np.clip(union_area, 1e-6, None)
    return iou


def convert_mmdet_preds_to_coco(pred_instance_list, score_thresh=0.0):
    coco_results = []
    for pred_dict in pred_instance_list:
        image_id = pred_dict['img_id']
        pred = pred_dict['pred_instances']

        labels = pred['labels'].tolist() if isinstance(pred['labels'], torch.Tensor) else pred['labels']
        bboxes = pred['bboxes'].tolist() if isinstance(pred['bboxes'], torch.Tensor) else pred['bboxes']
        scores = pred['scores'].tolist() if isinstance(pred['scores'], torch.Tensor) else pred['scores']
        labels = [label + 1 for label in labels]

        for label, box, score in zip(labels, bboxes, scores):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box
            coco_results.append({
                'image_id': int(image_id),
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': float(score),
                'category_id': int(label)
            })
    return coco_results


def compute_coco_eval_per_class(coco_gt, coco_dt, iou_thr=0.5):
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.iouThrs = np.array([iou_thr])
    coco_eval.evaluate()
    coco_eval.accumulate()

    class_ap50 = {}
    class_ar50 = {}
    class_auc50 = {}

    ap_list = []
    ar_list = []
    auc_list = []

    for idx, catId in enumerate(coco_gt.getCatIds()):
        # Precision shape: [T=1, R=101, K, A=1, M=1]
        precision = coco_eval.eval['precision'][0, :, idx, 0, 2]  # IoU=0.5, area=all, maxDet=100
        recall = coco_eval.params.recThrs
        recall_vals = coco_eval.eval['recall'][:, idx, 0, 2]

        valid = precision[precision > -1]
        ap = np.mean(valid) if valid.size > 0 else 0.0
        valid_recalls = recall_vals[recall_vals > -1]
        ar = float(np.mean(valid_recalls)) if valid_recalls.size > 0 else 0.0

        if np.all(precision == -1):
            auc = 0.0
        else:
            prec_valid = precision.copy()
            prec_valid[prec_valid == -1] = 0.0
            auc = np.trapz(prec_valid, recall)

        class_ap50[catId] = ap
        class_ar50[catId] = ar
        class_auc50[catId] = auc

        ap_list.append(ap)
        ar_list.append(ar)
        auc_list.append(auc)

    # 添加平均项
    class_ap50['average'] = np.mean(ap_list)
    class_ar50['average'] = np.mean(ar_list)
    class_auc50['average'] = np.mean(auc_list)

    return class_ap50, class_ar50, class_auc50




from collections import defaultdict

def compute_classwise_manual_metrics(coco_gt, coco_dt_list, iou_thr=0.5):
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for img_id in coco_gt.getImgIds():
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        dt_anns = [d for d in coco_dt_list if d['image_id'] == img_id]

        for cat_id in coco_gt.getCatIds():
            gt_boxes = [a['bbox'] for a in gt_anns if a['category_id'] == cat_id]
            pred_boxes = [d['bbox'] + [d['score']] for d in dt_anns if d['category_id'] == cat_id]

            gt_boxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in gt_boxes]
            used = [False] * len(gt_boxes_xyxy)

            for pred_box in pred_boxes:
                px1, py1, pw, ph, score = pred_box
                px2, py2 = px1 + pw, py1 + ph
                matched = False
                for i, gt_box in enumerate(gt_boxes_xyxy):
                    gx1, gy1, gx2, gy2 = gt_box
                    ix1 = max(px1, gx1)
                    iy1 = max(py1, gy1)
                    ix2 = min(px2, gx2)
                    iy2 = min(py2, gy2)
                    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
                    union = (px2 - px1) * (py2 - py1) + (gx2 - gx1) * (gy2 - gy1) - inter
                    iou = inter / union if union > 0 else 0
                    if iou >= iou_thr and not used[i]:
                        used[i] = True
                        matched = True
                        break
                if matched:
                    class_metrics[cat_id]['tp'] += 1
                else:
                    class_metrics[cat_id]['fp'] += 1
            class_metrics[cat_id]['fn'] += used.count(False)

    final_results = {}
    metric_sums = {
        'Precision': 0.0,
        'Recall': 0.0,
        'F1': 0.0,
        'Sensitivity': 0.0,
        'Specificity': 0.0
    }

    valid_classes = 0

    for cat_id, m in class_metrics.items():
        tp, fp, fn = m['tp'], m['fp'], m['fn']
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        sens = rec
        spec = 0.0  # TN 不定义，设为 0

        final_results[cat_id] = {
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'Sensitivity': sens,
            'Specificity': spec
        }

        metric_sums['Precision'] += prec
        metric_sums['Recall'] += rec
        metric_sums['F1'] += f1
        metric_sums['Sensitivity'] += sens
        metric_sums['Specificity'] += spec
        valid_classes += 1

    # 添加平均值
    if valid_classes > 0:
        final_results['average'] = {
            k: metric_sums[k] / valid_classes for k in metric_sums
        }

    return final_results


def load_pred_file(pred_path, score_thresh=0.05):
    if pred_path.endswith('.pth') or pred_path.endswith('.pt'):
        preds = torch.load(pred_path)
        preds = convert_mmdet_preds_to_coco(preds, score_thresh=score_thresh)

    elif pred_path.endswith('.json'):
        with open(pred_path, 'r') as f:
            preds = json.load(f)
        # 在 JSON 情况下手动筛选 score >= 阈值
        preds = [d for d in preds if d.get('score', 1.0) >= score_thresh]

    else:
        raise ValueError(f"Unsupported prediction file type: {pred_path}")
    
    return preds





def aggregate_overall_metrics(all_results):
    overall_keys = all_results[0].keys()
    summary = {}
    for key in overall_keys:
        values = [fold[key] for fold in all_results]
        summary[key] = {
            'mean': round(np.mean(values) * 100, 2),  # 转换为百分数
            'std': round(np.std(values) * 100, 2)}
    return summary

def aggregate_per_class_metrics_across_folds(all_folds_per_class_metrics):
    aggregated = defaultdict(lambda: defaultdict(list))

    # 遍历每一折的结果
    for fold_result in all_folds_per_class_metrics:
        for class_name, metrics in fold_result.items():
            for metric_name, value in metrics.items():
                aggregated[class_name][metric_name].append(value)

    # 计算每类每个指标的 mean 和 std
    final_result = {}
    for class_name, metrics in aggregated.items():
        final_result[class_name] = {}
        for metric_name, values in metrics.items():
            values_array = np.array(values)
            final_result[class_name][metric_name] = {
                'mean': round(np.mean(values_array) * 100, 2),  # 百分数
                'std': round(np.std(values_array) * 100, 2)
            }
    
    return final_result



def ensemble_preds_across_folds(fold_preds_list, coco_gt, method="wbf", iou_thr=0.63, skip_box_thr=0.0001, sigma=0.05, weights=None):
    from collections import defaultdict
    import numpy as np
    
    image_pred_dict = defaultdict(list)
    for fold_idx, fold_preds in enumerate(fold_preds_list):
        for pred in fold_preds:
            image_pred_dict[pred["image_id"]].append((fold_idx, pred))

    final_ensembled_preds = []

    for image_id, preds_with_folds in image_pred_dict.items():
        # 🔄 正确获取图像尺寸
        # import pdb;pdb.set_trace()
        img_info = coco_gt.loadImgs([image_id])[0]
        width, height = img_info['width'], img_info['height']

        n_folds = max(idx for idx, _ in preds_with_folds) + 1
        fold_boxes = [[] for _ in range(n_folds)]
        fold_scores = [[] for _ in range(n_folds)]
        fold_labels = [[] for _ in range(n_folds)]

        for fold_idx, pred in preds_with_folds:
            x, y, w, h = pred['bbox']
            x1, y1, x2, y2 = x / width, y / height, (x + w) / width, (y + h) / height
            fold_boxes[fold_idx].append([x1, y1, x2, y2])
            fold_scores[fold_idx].append(pred['score'])
            fold_labels[fold_idx].append(pred['category_id'])

        # Remove empty folds
        boxes_list = [b for b in fold_boxes if len(b) > 0]
        scores_list = [s for s in fold_scores if len(s) > 0]
        labels_list = [l for l in fold_labels if len(l) > 0]
        used_weights = weights if weights else [1.0] * len(boxes_list)

        if not boxes_list:
            continue
        # import pdb;pdb.set_trace()
        # 🧠 融合策略
        if method == "wbf":
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=used_weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
            )
        elif method == "nms":
            boxes, scores, labels = nms(
                boxes_list, scores_list, labels_list,
                weights=used_weights, iou_thr=iou_thr
            )
        elif method == "soft_nms":
            boxes, scores, labels = soft_nms(
                boxes_list, scores_list, labels_list,
                weights=used_weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr
            )
        elif method == "nmw":
            boxes, scores, labels = non_maximum_weighted(
                boxes_list, scores_list, labels_list,
                weights=used_weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
            )
        elif method == "avg":
            import numpy as np

            # flatten all boxes of same class
            flat_boxes = [b for sublist in boxes_list for b in sublist]
            flat_scores = [s for sublist in scores_list for s in sublist]
            flat_labels = [l for sublist in labels_list for l in sublist]

            boxes, scores, labels = [], [], []

            for class_id in set(flat_labels):
                cls_boxes = np.array([b for b, l in zip(flat_boxes, flat_labels) if l == class_id])
                cls_scores = np.array([s for s, l in zip(flat_scores, flat_labels) if l == class_id])

                if len(cls_boxes) == 0:
                    continue

                used = np.zeros(len(cls_boxes), dtype=bool)

                for i in range(len(cls_boxes)):
                    if used[i]:
                        continue
                    # 当前框作为参考中心
                    ref_box = cls_boxes[i]
                    ious = compute_iou_np(np.expand_dims(ref_box, 0), cls_boxes)[0]
                    group_idx = np.where((ious > iou_thr) & (~used))[0]

                    # 计算平均框（加权 or 不加权）
                    grouped_boxes = cls_boxes[group_idx]
                    grouped_scores = cls_scores[group_idx]
                    avg_box = np.average(grouped_boxes, axis=0, weights=grouped_scores)
                    avg_score = np.mean(grouped_scores)

                    boxes.append(avg_box.tolist())
                    scores.append(float(avg_score))
                    labels.append(class_id)
                    used[group_idx] = True
        elif method == "max_score":

            from torchvision.ops import nms as torchvision_nms

            flat_boxes = [b for sublist in boxes_list for b in sublist]
            flat_scores = [s for sublist in scores_list for s in sublist]
            flat_labels = [l for sublist in labels_list for l in sublist]

            boxes = []
            scores = []
            labels = []

            for class_id in set(flat_labels):
                cls_boxes = np.array([b for b, l in zip(flat_boxes, flat_labels) if l == class_id])
                cls_scores = np.array([s for s, l in zip(flat_scores, flat_labels) if l == class_id])

                if len(cls_boxes) == 0:
                    continue

                cls_boxes_tensor = torch.tensor(cls_boxes, dtype=torch.float32)
                cls_scores_tensor = torch.tensor(cls_scores, dtype=torch.float32)

                keep = torchvision_nms(cls_boxes_tensor, cls_scores_tensor, 0.5)
                keep = keep.numpy()

                boxes.extend(cls_boxes[keep])
                scores.extend(cls_scores[keep])
                labels.extend([class_id] * len(keep))
        elif method == "vote":
            # 多折出现次数 >=2 的框保留
            
            flat_boxes = [b for sublist in boxes_list for b in sublist]
            flat_scores = [s for sublist in scores_list for s in sublist]
            flat_labels = [l for sublist in labels_list for l in sublist]

            boxes, scores, labels = [], [], []

            for class_id in set(flat_labels):
                cls_boxes = np.array([b for b, l in zip(flat_boxes, flat_labels) if l == class_id])
                cls_scores = np.array([s for s, l in zip(flat_scores, flat_labels) if l == class_id])

                if len(cls_boxes) == 0:
                    continue

                used = np.zeros(len(cls_boxes), dtype=bool)

                for i in range(len(cls_boxes)):
                    if used[i]:
                        continue
                    ref_box = cls_boxes[i]
                    ious = compute_iou_np(np.expand_dims(ref_box, 0), cls_boxes)[0]
                    group_idx = np.where(ious > iou_thr)[0]

                    # 只保留被多个 box “投票”支持的目标
                    if len(group_idx) >= 2:
                        grouped_boxes = cls_boxes[group_idx]
                        grouped_scores = cls_scores[group_idx]
                        avg_box = np.average(grouped_boxes, axis=0, weights=grouped_scores)
                        avg_score = np.mean(grouped_scores)

                        boxes.append(avg_box.tolist())
                        scores.append(float(avg_score))
                        labels.append(class_id)

                    used[group_idx] = True
        else:
            raise ValueError(f"Unsupported ensemble method: {method}")

        boxes = np.array(boxes)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            final_ensembled_preds.append({
                "image_id": int(image_id),
                "bbox": [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height],
                "score": float(score),
                "category_id": int(label)
            })

    return final_ensembled_preds

# def evaluate_all_folds(base_dir, n_folds=5, ensemble_method="wbf", iou_thr=0.5, weights=None):
#     gt_file = "DATASET/SurginDataset/annotations/instances_test.json"
#     all_fold_preds = []
#     coco_gt = COCO(gt_file)
    
    
    
#     for i in range(n_folds):
#         for j in range(5):
#         pred_file = os.path.join(base_dir, f"ft_{i+1}", "eval/model_best/inference/test/bbox.json")
#         preds = load_pred_file(pred_file)
#         all_fold_preds.append(preds)

def evaluate_all_folds(base_dir, n_folds=5, ensemble_method="wbf", iou_thr=0.5):
    gt_file = f"DATASET/SurginDataset/test/instances_test.json"
    all_fold_preds = []
    coco_gt = COCO(gt_file)
    weights = [1]*n_folds
    for i in range(n_folds):
        
        pred_file = os.path.join(base_dir, f"ft_{i+1}", "eval/model_best/inference/test/bbox4.json")
        preds = load_pred_file(pred_file)
        all_fold_preds.append(preds)

    # 执行融合
    ensembled_preds = ensemble_preds_across_folds(
        all_fold_preds,
        coco_gt,
        method=ensemble_method,
        weights=weights
    )
    # import pdb;pdb.set_trace()

    coco_dt = coco_gt.loadRes(ensembled_preds)
    ensembled_pred_05 = [ensembled_pred for ensembled_pred in ensembled_preds if ensembled_pred['score']>0.005]


    ap50_dict, ar50_dict, auc50_dict = compute_coco_eval_per_class(coco_gt, coco_dt, iou_thr=iou_thr)
    manual_metrics = compute_classwise_manual_metrics(coco_gt, ensembled_pred_05, iou_thr=iou_thr)
    
    # import pdb;pdb.set_trace()
    if "nms" in ensemble_method:
        ensembled_pred_30 = [ensembled_pred for ensembled_pred in ensembled_preds if ensembled_pred['score']>0.12]
    else:
        ensembled_pred_30 = [ensembled_pred for ensembled_pred in ensembled_preds if ensembled_pred['score']>0.45]
    manual_metrics_30 = compute_classwise_manual_metrics(coco_gt, ensembled_pred_30, iou_thr=iou_thr)


    # import pdb;pdb.set_trace()
    # 构建 per-class 汇总
    per_class_metrics = {}
    for cat_id in coco_gt.getCatIds():
        name = coco_gt.loadCats([cat_id])[0]['name']
        per_class_metrics[name] = {
            'AP': ap50_dict.get(cat_id, 0),
            'AUC': auc50_dict.get(cat_id, 0),
            "Recall" : manual_metrics.get(cat_id,)['Recall'],
            "Precision": manual_metrics_30.get(cat_id,)['Precision'],
            "F1": manual_metrics_30.get(cat_id,)['F1'],
        }
    cat_id = 'average'
    per_class_metrics[cat_id] =  {
            'AP': ap50_dict.get(cat_id, 0),
            'AUC': auc50_dict.get(cat_id, 0),
            "Recall" : manual_metrics.get(cat_id,)['Recall'],
            "Precision": manual_metrics_30.get(cat_id,)['Precision'],
            "F1": manual_metrics_30.get(cat_id,)['F1'],
        }   
    # import pdb;pdb.set_trace()

    # overall metrics
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.iouThrs = np.array([iou_thr])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap50 = float(coco_eval.stats[0])
    ar50 = float(coco_eval.stats[8])
    overall_metrics = {
        "AP50": ap50,
        "AR50": ar50,
    }

    # 用于统一结构
    return overall_metrics, per_class_metrics

def print_metrics(overall_metrics, per_class_metrics):
    # Overall Metrics
    # import pdb;pdb.set_trace()
    print("=== Overall Metrics ===")
    print(f"{'Metric':<10} | {'Mean (%)':>8}")
    print("-" * 23)
    for k, v in overall_metrics.items():
        print(f"{k:<10} | {v*100:.2f}")

    # Per-Class Metrics
    print("\n=== Per-Class Metrics ===")
    headers = [
        "Class", "AP (%)", "AUC (%)", "Precision (%)",
        "Recall (%)", "F1 (%)",
    ]
    col_widths = [32, 8, 8, 14, 12, 8, 16, 16]
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for cls, m in per_class_metrics.items():
        row = [
            cls,
            f"{m['AP']*100:.2f}",
            f"{m['AUC']*100:.2f}",
            f"{m['Precision']*100:.2f}",
            f"{m['Recall']*100:.2f}",
            f"{m['F1']*100:.2f}",
        ]
        print(" | ".join(f"{val:<{w}}" for val, w in zip(row, col_widths)))



if __name__ == "__main__":
    base_path = "OUTPUT/wound"
    
    overall_metrics, per_class_metrics = evaluate_all_folds(
    base_dir=base_path,
    n_folds=5,iou_thr=0.5,
    ensemble_method="wbf")  #nms soft_nms wbf nmw  avg  max_score vote
    print_metrics(overall_metrics=overall_metrics, per_class_metrics=per_class_metrics)




            
