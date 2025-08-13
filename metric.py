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

    n_categories = len(coco_gt.getCatIds())
    class_ap50 = {}
    class_ar50 = {}
    class_auc50 = {}

    for idx, catId in enumerate(coco_gt.getCatIds()):
        # Precision shape: [T=1, R=101, K, A=1, M=1]
        precision = coco_eval.eval['precision'][0, :, idx, 0, 2]  # IoU=0.5, area=all, maxDet=100
        recall = coco_eval.params.recThrs  # 101 recall thresholds
        # import pdb;pdb.set_trace()
        recall_vals = coco_eval.eval['recall'][:, idx, 0, 2]


        valid = precision[precision > -1]
        ap = np.mean(valid) if valid.size > 0 else 0.0
        valid_recalls = recall_vals[recall_vals > -1]
        ar = float(np.mean(valid_recalls)) if valid_recalls.size > 0 else 0.0

        # Compute AUC (area under PR curve): trapezoidal rule
        if np.all(precision == -1):
            auc = 0.0
        else:
            prec_valid = precision.copy()
            prec_valid[prec_valid == -1] = 0.0  # replace -1 with 0 for integration
            auc = np.trapz(prec_valid, recall)

        class_ap50[catId] = ap
        class_ar50[catId] = ar
        class_auc50[catId] = auc

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

    # # 添加平均值
    # if valid_classes > 0:
    #     final_results['average'] = {
    #         k: metric_sums[k] / valid_classes for k in metric_sums
    #     }

    return final_results


def compute_detection_metrics(gt_path, pred_path, iou_thr=0.5):
    coco_gt = COCO(gt_path)
    # with open(pred_path, 'r') as f:
    #     preds = json.load(f)  # 改为json读取
    # import pdb;pdb.set_trace()
    preds = load_pred_file(pred_path)
    coco_dt = coco_gt.loadRes(preds)
    

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.iouThrs = np.array([iou_thr])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap50 = float(coco_eval.stats[0])  # AP at IoU=0.5
    ar50 = float(coco_eval.stats[8])  # AR at IoU=0.5


    return {
        "AP50": ap50,
        "AR50": ar50,
    }

def evaluate_per_class(gt_path, pred_path):
    coco_dt_list = load_pred_file(pred_path)
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(coco_dt_list)
    
    coco_dt_list_pred_05 = [coco_dt for coco_dt in coco_dt_list if coco_dt['score']>0.005]
        
    ap50_dict, ar50_dict, auc50_dict = compute_coco_eval_per_class(coco_gt, coco_dt, iou_thr=0.5)
    manual_metrics = compute_classwise_manual_metrics(coco_gt, coco_dt_list_pred_05, iou_thr=0.5)
    coco_dt_list_pred_45 = [coco_dt for coco_dt in coco_dt_list if coco_dt['score']>0.45]
    manual_metrics_45 = compute_classwise_manual_metrics(coco_gt, coco_dt_list_pred_45, iou_thr=0.5)

    per_class_metrics = {}
    for cat_id in coco_gt.getCatIds():
        name = coco_gt.loadCats([cat_id])[0]['name']
        per_class_metrics[name] = {
            'AP': ap50_dict.get(cat_id, 0),
            'AUC': auc50_dict.get(cat_id, 0),
            "Precision": manual_metrics_45.get(cat_id,)['Precision'],
            "Recall" : manual_metrics.get(cat_id,)['Recall'],           
            "F1": manual_metrics_45.get(cat_id,)['F1'],
        }
 
    return per_class_metrics

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


def evaluate_all_folds(base_dir, n_folds=5):
    all_folds_per_class_metrics = []
    all_results = []

    for i in range(n_folds):
        gt_file = f"DATASET/SurginDataset/annotations/instances_test.json"
        gt_file = f"DATASET/SurginDataset/test/instances_test.json"
        pred_file = os.path.join(base_dir, f"ft_{i+1}", "eval/model_best/inference/test/bbox4.json")
        metrics = compute_detection_metrics(gt_file, pred_file)
        per_class_metrics = evaluate_per_class(gt_file, pred_file)
        # print(f"Fold {i} per class metrics:", per_class_metrics)
        
        all_folds_per_class_metrics.append(per_class_metrics)
        all_results.append(metrics)
        
    overall_summary = aggregate_overall_metrics(all_results)
    per_class_summary = aggregate_per_class_metrics_across_folds(all_folds_per_class_metrics)
    
    return overall_summary, per_class_summary


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


def save_overall_metrics_to_csv(overall_summary, save_path="overall_metrics.csv"):
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean (%)", "Std (%)"])
        for metric, values in overall_summary.items():
            writer.writerow([metric, values['mean'], values['std']])
    print(f"Saved overall metrics to {save_path}")

def save_per_class_metrics_to_csv(per_class_summary, save_path="per_class_metrics.csv"):
    metric_names = next(iter(per_class_summary.values())).keys()
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["Class", "Metric", "Mean (%)", "Std (%)"]
        writer.writerow(header)
        for class_name, metrics in per_class_summary.items():
            for metric, values in metrics.items():
                writer.writerow([class_name, metric, values['mean'], values['std']])
    print(f"Saved per-class metrics to {save_path}")

if __name__ == "__main__":
    base_path = "OUTPUT/wound"
    overall_summary, per_class_summary = evaluate_all_folds(base_path, n_folds=5)

    print("=== Overall Metrics ===")
    for k, v in overall_summary.items():
        print(f"{k}: mean = {v['mean']}, std = {v['std']}")

    aggregate_metrics = {}
    print("\n=== Per-Class Metrics ===")
    for cls, metrics in per_class_summary.items():
        print(f"\nClass: {cls}")
        for metric, val in metrics.items():
            print(f"{metric:>12}: mean = {val['mean']}, std = {val['std']}")
            if metric not in aggregate_metrics:
                aggregate_metrics[metric] = []
            aggregate_metrics[metric].append(val['mean'])
            
            
    print("\n=== Overall Macro Mean ===")
    for metric, values in aggregate_metrics.items():
        macro_mean = sum(values) / len(values)
        print(f"{metric:>12}: {macro_mean:.2f}")