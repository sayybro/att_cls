import numpy as np
from collections import defaultdict
import copy
import json

class VAWEvaluator():
    def __init__(self, preds, gts, subject_category_id, rare_triplets,non_rare_triplets, valid_masks, max_pred):
        self.overlap_iou = 0.5
        self.max_attrs = max_pred
        self.rare_triplets = rare_triplets
        self.non_rare_triplets = non_rare_triplets
        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_double = []
        self.valid_masks = valid_masks
        self.preds = []
        self.ggg = gts
        self.gts = []
        
        for img_gts in gts:
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k not in ['id','type','dataset'] }
            self.gts.append({'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_gts['boxes'], img_gts['labels'])]
                            ,'attr_annotation':[]})

            for i,attr in enumerate(img_gts['pos_att_classes']): 
                attr_idxs = np.nonzero(attr==1)[0]
                for j in attr_idxs:
                    if self.valid_masks[j]==1: 
                        self.gts[-1]['attr_annotation'].append({'object_id': i, 'category_id': j}) 
                    
            for attr in self.gts[-1]['attr_annotation']:
                double = (attr['category_id']) 

                if double not in self.gt_double:
                    self.gt_double.append(double) 

                self.sum_gts[double] += 1 

        for i, img_preds in enumerate(preds):
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items() if k != 'att_recognition_time'}
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            attr_scores = img_preds['attr_scores']
            attr_labels = np.tile(np.arange(attr_scores.shape[1]), (attr_scores.shape[0], 1))
            object_ids = np.tile(img_preds['obj_ids'], (attr_scores.shape[1], 1)).T
            attr_scores = attr_scores.ravel()
            attr_labels = attr_labels.ravel()
            object_ids = object_ids.ravel()
            masks = valid_masks[attr_labels]
            attr_scores *= masks
            attrs = [{'object_id': object_id, 'category_id': category_id, 'score': score} for
                    object_id, category_id, score in zip(object_ids, attr_labels, attr_scores)]   
            attrs.sort(key=lambda k: (k.get('score', 0)), reverse=True)
            attrs = attrs[:self.max_attrs]
            self.preds.append({
                'predictions': bboxes, 
                'attr_prediction': attrs 
            })
    def evaluate(self):
        count_dict = dict()

        for img_id, (img_preds, img_gts) in enumerate(zip(self.preds, self.gts)):
            if img_gts['attr_annotation']:
                for annotation in img_gts['attr_annotation']:
                    attribute_idx = str(annotation['category_id'])
                    if attribute_idx in count_dict.keys():
                        count_dict[attribute_idx] += 1
                    else:
                        count_dict[attribute_idx] = 1
            print(f"Evaluating Score Matrix... : [{(img_id+1):>4}/{len(self.gts):<4}]" ,flush=True, end="\r")
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations'] 
            pred_attrs = img_preds['attr_prediction'] 
            gt_attrs = img_gts['attr_annotation'] 
            if len(gt_bboxes) != 0:
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_attrs, gt_attrs, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_attr in pred_attrs:
                    double = [pred_attr['category_id']]
                    if double not in self.gt_double:
                        continue
                    self.tp[double].append(0)
                    self.fp[double].append(1)
                    self.score[double].append(pred_attr['score'])

        with open('count_dict.json', 'w') as f:
            json.dump(count_dict, f)
        print(f"[stats] Score Matrix Generation completed!!")
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        for double in self.gt_double:
            sum_gts = self.sum_gts[double]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[double]))
            fp = np.array((self.fp[double]))
            if len(tp) == 0:
                ap[double] = 0
                max_recall[double] = 0
                if double in self.rare_triplets:
                    rare_ap[double] = 0
                elif double in self.non_rare_triplets:
                    non_rare_ap[double] = 0
                else:
                    print('Warning: triplet {} is neither in rare double nor in non-rare double'.format(double))
                continue

            score = np.array(self.score[double])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[double] = self.voc_ap(rec, prec)
            max_recall[double] = np.amax(rec)
            if double in self.rare_triplets:
                rare_ap[double] = ap[double]
            elif double in self.non_rare_triplets:
                non_rare_ap[double] = ap[double]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(double))
        
        m_ap = np.mean(list(ap.values()))
        m_ap_rare = np.mean(list(rare_ap.values()))
        m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))

        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare, m_max_recall))
        print('--------------------')

        return {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare, 'mean max recall': m_max_recall}

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_attrs, gt_attrs, match_pairs, pred_bboxes, bbox_overlaps):
        pos_pred_ids = match_pairs.keys() 
        vis_tag = np.zeros(len(gt_attrs)) 
        pred_attrs.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_attrs) != 0:
            for pred_attr in pred_attrs:           
                is_match = 0

                if len(match_pairs) != 0 and pred_attr['object_id'] in pos_pred_ids:
                    pred_obj_ids = match_pairs[pred_attr['object_id']]                    
                    pred_obj_overlaps = bbox_overlaps[pred_attr['object_id']] 
                    pred_category_id = pred_attr['category_id']
                    max_overlap = 0
                    max_gt_attr = 0

                    for gt_attr in gt_attrs: 

                        if gt_attr['object_id'] in pred_obj_ids and pred_category_id == gt_attr['category_id']:
                            is_match = 1

                            min_overlap_gt = pred_obj_overlaps[pred_obj_ids.index(gt_attr['object_id'])]
 
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_attr = gt_attr
                
          
                double = (pred_attr['category_id'])          
                if double not in self.gt_double:
                    continue
                
                if is_match == 1 and vis_tag[gt_attrs.index(max_gt_attr)] == 0:
                    self.fp[double].append(0) 
                    self.tp[double].append(1) 
                    vis_tag[gt_attrs.index(max_gt_attr)] = 1
                
                else:
                    self.fp[double].append(1) 
                    self.tp[double].append(0) 
                self.score[double].append(pred_attr['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1): 
            for j, bbox2 in enumerate(bbox_list2): 
                iou_i = self.compute_IOU(bbox1, bbox2) 
                iou_mat[i, j] = iou_i


        iou_mat_ov=iou_mat.copy()

    
        iou_mat[iou_mat>=self.overlap_iou] = 1
        iou_mat[iou_mat<self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]): 
                if pred_id not in match_pairs_dict.keys():  
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i]) 
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id]) 
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))

        rec1 = bbox1['bbox']
        rec2 = bbox2['bbox']

        S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
        S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)


        sum_area = S_rec1 + S_rec2


        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
            return intersect / (sum_area - intersect)
