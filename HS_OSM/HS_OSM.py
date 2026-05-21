import os
import numpy as np
import torch
import logging
from collections import OrderedDict
import copy
import itertools
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
import json

# Import existing evaluator and S-measure components
from detectron2.evaluation import SemSegEvaluator
from SS_mIoU.Hierarchy_aware_similarity import HierarchicalSimilarity
from SS_mIoU.s_object import s_object
from SS_mIoU.s_region import s_region


def normalize_prob_map(prob_map):
    """Normalize a probability map to the range [0, 1]."""
    min_val = np.min(prob_map)
    max_val = np.max(prob_map)
    if max_val > min_val:
        return (prob_map - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(prob_map)  # Return a zero map if the map is constant

class Semantic_Structure_Evaluator(SemSegEvaluator):
    """
    A semantic segmentation evaluator that combines hierarchy-aware similarity 
    and structured metrics (S-measure).
    """
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
        gamma=0.3,
        beta=0.5,
        delta=4,
        alpha=0.4,
        glove_path='glove.6B.300d.txt',
        precomputed_matrix_path=None,
        s_alpha=0.5,  # Weight parameter for S-measure
        glove_mapping_file=None,
        wordnet_mapping_file=None,
        use_which_adj = 0,
        soft_s_topk = 3
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )        

        # Get class names from metadata
        self.class_names = self._class_names
        
        # Set S-measure parameters
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.s_alpha = s_alpha
        self.soft_s_topk = soft_s_topk

        # Initialize lists for "soft" S-measure
        self._soft_s_measures = []
        self._soft_s_object_overall = []
        self._soft_s_region_overall = []
        self._class_soft_s_objects = {}
        self._class_soft_s_regions = {}

        # Initialize lists for "hard" S-measure
        self._hard_s_measures = []
        self._hard_s_object_overall = []
        self._hard_s_region_overall = []
        self._class_hard_s_objects = {}
        self._class_hard_s_regions = {}

        # Initialize similarity matrix
        if precomputed_matrix_path and os.path.exists(precomputed_matrix_path):
            # Load precomputed similarity matrix
            import pandas as pd
            df_sim = pd.read_csv(precomputed_matrix_path, index_col=0)
            self.simi_matrix = df_sim.values
        else:
            # Initialize similarity calculator
            self.sim_calculator = HierarchicalSimilarity(
                gamma=gamma, 
                beta=beta, 
                delta=delta, 
                alpha=alpha,  # Weight for modifier impact
                glove_path=glove_path,
                glove_mapping_file=glove_mapping_file,
                wordnet_mapping_file=wordnet_mapping_file,
                use_which_adj=use_which_adj
            )
            
            # Compute the inter-class similarity matrix
            self.simi_matrix = self._compute_similarity_matrix()
            
            # If a path is provided, save the similarity matrix
            if precomputed_matrix_path:
                import pandas as pd
                df_sim = pd.DataFrame(
                    self.simi_matrix, 
                    index=self.class_names, 
                    columns=self.class_names
                )
                os.makedirs(os.path.dirname(precomputed_matrix_path), exist_ok=True)
                df_sim.to_csv(precomputed_matrix_path)
        
        # Convert similarity matrix to a torch tensor
        self._similarity_tensor = torch.from_numpy(self.simi_matrix).float()

    def _compute_similarity_matrix(self):
        """
        Computes the hierarchy-aware similarity matrix between classes.
        """
        n = len(self.class_names)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            sim_matrix[i, i] = 1.0  # Diagonal is 1.0
            for j in range(i + 1, n):
                # Get class names
                class_i = self.class_names[i]
                class_j = self.class_names[j]
                
                # Calculate similarity
                sim_value = self.sim_calculator.get_sim(class_i, class_j)
                
                # If the return value is a tuple (similarity, base_similarity), take the first element
                if isinstance(sim_value, tuple):
                    sim_value = sim_value[0]
                    
                sim_matrix[i, j] = sim_value
                sim_matrix[j, i] = sim_value  # Symmetric matrix
                
        return sim_matrix 
        
    def process(self, inputs, outputs):
        """
        Process model outputs to compute confusion matrix and record per-image IoU and S-measure.
        """
        # Ensure output directory exists
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file = os.path.join(self._output_dir, "per_image_metrics.txt")
            fn = open(file, 'a')

        for input, output in zip(inputs, outputs):
            
            # Clone original logits for S-measure calculation
            sem_seg_logits = output["sem_seg"].clone()
            
            processed_output = self.post_process_func(
                output["sem_seg"], image=np.array(Image.open(input["file_name"]))
            )
            output = processed_output.argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)

            with PathManager.open(
                self.input_file_to_gt_file[input["file_name"]], "rb"
            ) as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            # Update the global confusion matrix
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            # Per-image confusion matrix
            cur_conf_matrix = np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)
            cur_conf_matrix = cur_conf_matrix.astype(np.float64)

            # Calculate per-image hierarchical IoU (h_mIoU)
            valid_part = cur_conf_matrix[:-1, :-1]
            if self.simi_matrix is not None and self.simi_matrix.shape[0] == valid_part.shape[0]:
                cur_ow_conf_matrix = copy.deepcopy(cur_conf_matrix)
                valid_part = valid_part.astype(np.float64)
                ow_tp = np.sum(valid_part * self.simi_matrix, axis=0)
                valid_part_copy = valid_part * (1 - self.simi_matrix)
                row, col = np.diag_indices_from(valid_part_copy)
                valid_part_copy[row, col] = ow_tp
                cur_ow_conf_matrix = cur_ow_conf_matrix.astype(np.float64)
                cur_ow_conf_matrix[:-1, :-1] = valid_part_copy
            else:
                cur_ow_conf_matrix = copy.deepcopy(cur_conf_matrix)

            # Calculate standard evaluation metrics for the current image
            per_acc = np.full(self._num_classes, np.nan, dtype=float)
            per_iou = np.full(self._num_classes, np.nan, dtype=float)
            per_tp = cur_conf_matrix.diagonal()[:-1].astype(float)
            per_pos_gt = cur_conf_matrix[:-1, :-1].sum(axis=0).astype(float)
            class_weights_iou = per_pos_gt / np.sum(per_pos_gt)
            per_pos_pred = cur_conf_matrix[:-1, :-1].sum(axis=1).astype(float)
            per_acc_valid = per_pos_gt > 0
            per_acc[per_acc_valid] = per_tp[per_acc_valid] / per_pos_gt[per_acc_valid]
            per_union = per_pos_gt + per_pos_pred - per_tp
            per_iou_valid = np.logical_and(per_acc_valid, per_union > 0)
            per_iou[per_iou_valid] = per_tp[per_iou_valid] / per_union[per_iou_valid]
            per_macc = np.sum(per_acc[per_acc_valid]) / np.sum(per_acc_valid)
            per_miou = np.sum(per_iou[per_iou_valid]) / np.sum(per_iou_valid)
            per_fiou = np.sum(per_iou[per_iou_valid] * class_weights_iou[per_iou_valid])
            per_pacc = np.sum(per_tp) / np.sum(per_pos_gt)

            # Calculate hierarchical evaluation metrics for the current image
            per_h_acc = np.full(self._num_classes, np.nan, dtype=float)
            per_h_iou = np.full(self._num_classes, np.nan, dtype=float)
            per_h_tp = cur_ow_conf_matrix.diagonal()[:-1].astype(float)
            per_h_pos_gt = cur_ow_conf_matrix[:-1, :-1].sum(axis=0).astype(float)
            h_class_weights = per_h_pos_gt / np.sum(per_h_pos_gt)
            per_h_pos_pred = cur_ow_conf_matrix[:-1, :-1].sum(axis=1).astype(float)
            per_h_acc_valid = per_h_pos_gt > 0
            per_h_acc[per_h_acc_valid] = per_h_tp[per_h_acc_valid] / per_h_pos_gt[per_h_acc_valid]
            per_h_union = per_h_pos_gt + per_h_pos_pred - per_h_tp
            per_h_iou_valid = np.logical_and(per_h_acc_valid, per_h_union > 0)
            per_h_iou[per_h_iou_valid] = per_h_tp[per_h_iou_valid] / per_h_union[per_h_iou_valid]
            per_h_macc = np.sum(per_h_acc[per_h_acc_valid]) / np.sum(per_h_acc_valid)
            per_h_miou = np.sum(per_h_iou[per_h_iou_valid]) / np.sum(per_h_iou_valid)
            per_h_fiou = np.sum(per_h_iou[per_h_iou_valid] * h_class_weights[per_h_iou_valid])
            per_h_pacc = np.sum(per_h_tp) / np.sum(per_h_pos_gt)

            # --- S-measure Calculation ---
            sem_seg_probs = torch.softmax(sem_seg_logits, dim=0)
            present_classes = np.unique(gt)
            valid_present_classes = [c for c in present_classes if c != self._num_classes]
            total_class_pixels = np.sum([np.sum(gt == c) for c in valid_present_classes])

            # Initialize temporary storage for the current image
            class_weights = []
            # Soft S-measure
            soft_class_s_objects, soft_class_s_regions = [], []
            curr_soft_class_s_objects, curr_soft_class_s_regions = {}, {}
            # Hard S-measure
            hard_class_s_objects, hard_class_s_regions = [], []
            curr_hard_class_s_objects, curr_hard_class_s_regions = {}, {}

            # Calculate hard and soft S-measure for each present class
            for c_gt in valid_present_classes:
                gt_mask = (gt == c_gt)
                if np.sum(gt_mask) == 0:
                    continue
                
                class_weight = np.sum(gt_mask) / total_class_pixels
                class_weights.append(class_weight)
                
                probs_np = sem_seg_probs.cpu().numpy()

                # --- 1. Calculate Hard S-measure ---
                prob_map_hard = probs_np[c_gt]
                mask_hard = (pred == c_gt)
                
                masked_prob_map_hard = prob_map_hard.copy()
                masked_prob_map_hard[~mask_hard] = 0
                
                normalized_prob_map_hard = normalize_prob_map(masked_prob_map_hard)
                normalized_prob_map_hard = np.clip(normalized_prob_map_hard, 0, 1)
                
                s_obj_score_hard = s_object(normalized_prob_map_hard, gt_mask)
                s_reg_score_hard = s_region(normalized_prob_map_hard, gt_mask)

                if np.isnan(s_obj_score_hard) or s_obj_score_hard < 0: s_obj_score_hard = 0.0
                if np.isnan(s_reg_score_hard) or s_reg_score_hard < 0: s_reg_score_hard = 0.0
                
                curr_hard_class_s_objects[c_gt] = s_obj_score_hard
                curr_hard_class_s_regions[c_gt] = s_reg_score_hard
                hard_class_s_objects.append(s_obj_score_hard)
                hard_class_s_regions.append(s_reg_score_hard)

                # --- 2. Calculate Soft S-measure ---
                top_k = self.soft_s_topk
                sim_scores = self.simi_matrix[c_gt, :]
                k_to_use = min(top_k, len(sim_scores))
                top_k_indices = np.argpartition(-sim_scores, k_to_use - 1)[:k_to_use]
                tolerant_mask = np.isin(pred, top_k_indices)
                
                # Build a tolerant ground truth mask
                tolerant_gt_mask = np.zeros_like(gt, dtype=bool)
                for idx in top_k_indices:
                    tolerant_gt_mask = tolerant_gt_mask | (gt == idx)

                ideal_prob_map = np.zeros_like(pred, dtype=np.float64)
                if np.any(tolerant_mask):
                    pred_indices = pred[tolerant_mask]
                    pred_probs = probs_np[pred_indices, np.where(tolerant_mask)[0], np.where(tolerant_mask)[1]]
                    similarities = self.simi_matrix[c_gt, pred_indices]
                    ideal_values = pred_probs * similarities
                    ideal_prob_map[tolerant_mask] = ideal_values

                normalized_prob_map_soft = normalize_prob_map(ideal_prob_map) 
                normalized_prob_map_soft = np.clip(normalized_prob_map_soft, 0, 1)
                
                # Use the tolerant GT mask to calculate S-measure scores
                s_obj_score_soft = s_object(normalized_prob_map_soft, tolerant_gt_mask)
                s_reg_score_soft = s_region(normalized_prob_map_soft, tolerant_gt_mask)
                
                if np.isnan(s_obj_score_soft) or s_obj_score_soft < 0: s_obj_score_soft = 0.0
                if np.isnan(s_reg_score_soft) or s_reg_score_soft < 0: s_reg_score_soft = 0.0
                
                curr_soft_class_s_objects[c_gt] = s_obj_score_soft
                curr_soft_class_s_regions[c_gt] = s_reg_score_soft
                soft_class_s_objects.append(s_obj_score_soft)
                soft_class_s_regions.append(s_reg_score_soft)

            # --- Aggregate S-measure for the current image ---
            img_avg_soft_s_measure = 0
            img_avg_hard_s_measure = 0
            
            if class_weights:
                # Aggregate Soft S-measure
                weighted_s_object_soft = sum(s_obj * w for s_obj, w in zip(soft_class_s_objects, class_weights))
                weighted_s_region_soft = sum(s_reg * w for s_reg, w in zip(soft_class_s_regions, class_weights))
                img_avg_soft_s_measure = self.s_alpha * weighted_s_object_soft + (1 - self.s_alpha) * weighted_s_region_soft
                self._soft_s_measures.append(img_avg_soft_s_measure)
                self._soft_s_object_overall.append(weighted_s_object_soft)
                self._soft_s_region_overall.append(weighted_s_region_soft)
                for c, s in curr_soft_class_s_objects.items():
                    self._class_soft_s_objects.setdefault(c, []).append(s)
                for c, s in curr_soft_class_s_regions.items():
                    self._class_soft_s_regions.setdefault(c, []).append(s)

                # Aggregate Hard S-measure
                weighted_s_object_hard = sum(s_obj * w for s_obj, w in zip(hard_class_s_objects, class_weights))
                weighted_s_region_hard = sum(s_reg * w for s_reg, w in zip(hard_class_s_regions, class_weights))
                img_avg_hard_s_measure = self.s_alpha * weighted_s_object_hard + (1 - self.s_alpha) * weighted_s_region_hard
                self._hard_s_measures.append(img_avg_hard_s_measure)
                self._hard_s_object_overall.append(weighted_s_object_hard)
                self._hard_s_region_overall.append(weighted_s_region_hard)
                for c, s in curr_hard_class_s_objects.items():
                    self._class_hard_s_objects.setdefault(c, []).append(s)
                for c, s in curr_hard_class_s_regions.items():
                    self._class_hard_s_regions.setdefault(c, []).append(s)
            else:
                self._soft_s_measures.append(0)
                self._soft_s_object_overall.append(0)
                self._soft_s_region_overall.append(0)
                self._hard_s_measures.append(0)
                self._hard_s_object_overall.append(0)
                self._hard_s_region_overall.append(0)

            if self._output_dir:
                fn.write(input["file_name"] + '\n' + 
                            "mIoU\t" + str(per_miou) + '\t' +
                            "macc\t" + str(per_macc) + '\t' +
                            "fiou\t" + str(per_fiou) + '\t' +
                            "pacc\t" + str(per_pacc) + '\n' +
                            "h_mIoU\t" + str(per_h_miou) + '\t' +
                            "h_macc\t" + str(per_h_macc) + '\t' +
                            "h_fiou\t" + str(per_h_fiou) + '\t' +
                            "h_pacc\t" + str(per_h_pacc) + '\n' +
                            "Hard-S-measure\t" + str(img_avg_hard_s_measure) + '\n' +
                            "Soft-S-measure\t" + str(img_avg_soft_s_measure) + '\n'
                        )
            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
        if self._output_dir:
            fn.close()

    def evaluate(self):
        """
        Evaluate semantic segmentation metrics using hierarchy-aware similarity and structured measures.
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._soft_s_measures = list(itertools.chain(*all_gather(self._soft_s_measures)))
            self._soft_s_object_overall = list(itertools.chain(*all_gather(self._soft_s_object_overall)))
            self._soft_s_region_overall = list(itertools.chain(*all_gather(self._soft_s_region_overall)))
            self._hard_s_measures = list(itertools.chain(*all_gather(self._hard_s_measures)))
            self._hard_s_object_overall = list(itertools.chain(*all_gather(self._hard_s_object_overall)))
            self._hard_s_region_overall = list(itertools.chain(*all_gather(self._hard_s_region_overall)))
            
            # Gather dictionaries from all processes
            class_soft_s_objects_list = all_gather(self._class_soft_s_objects)
            class_soft_s_regions_list = all_gather(self._class_soft_s_regions)
            class_hard_s_objects_list = all_gather(self._class_hard_s_objects)
            class_hard_s_regions_list = all_gather(self._class_hard_s_regions)

            # Merge dictionaries
            self._class_soft_s_objects = {}
            self._class_soft_s_regions = {}
            self._class_hard_s_objects = {}
            self._class_hard_s_regions = {}

            for D in class_soft_s_objects_list:
                for k, v in D.items(): self._class_soft_s_objects.setdefault(k, []).extend(v)
            for D in class_soft_s_regions_list:
                for k, v in D.items(): self._class_soft_s_regions.setdefault(k, []).extend(v)
            for D in class_hard_s_objects_list:
                for k, v in D.items(): self._class_hard_s_objects.setdefault(k, []).extend(v)
            for D in class_hard_s_regions_list:
                for k, v in D.items(): self._class_hard_s_regions.setdefault(k, []).extend(v)

            self._predictions = list(itertools.chain(*self._predictions))
            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix
        
        self._conf_matrix_miou = self._conf_matrix.copy()
        if self.simi_matrix is not None:
            valid_part = self._conf_matrix[:-1, :-1]
            if self.simi_matrix.shape[0] == valid_part.shape[0]:
                valid_part = valid_part.astype(np.float64)
                ow_tp = np.sum(valid_part * self.simi_matrix, axis=0)
                valid_part *= (1 - self.simi_matrix)
                row, col = np.diag_indices_from(valid_part)
                valid_part[row, col] = ow_tp
                self._conf_matrix = self._conf_matrix.astype(np.float64)
                self._conf_matrix[:-1, :-1] = valid_part
            else:
                self._logger.warning(
                    f"Similarity matrix shape {self.simi_matrix.shape} does not match "
                    f"the confusion matrix's valid part {valid_part.shape}"
                )

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))
        
        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix_miou.diagonal()[:-1].astype(float)
        pos_gt = self._conf_matrix_miou[:-1, :-1].sum(axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = self._conf_matrix_miou[:-1, :-1].sum(axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)


        h_acc = np.full(self._num_classes, np.nan, dtype=float)
        h_iou = np.full(self._num_classes, np.nan, dtype=float)
        h_tp = self._conf_matrix.diagonal()[:-1].astype(float)
        h_pos_gt = self._conf_matrix[:-1, :-1].sum(axis=0).astype(float)
        h_class_weights = h_pos_gt / np.sum(h_pos_gt)
        h_pos_pred = self._conf_matrix[:-1, :-1].sum(axis=1).astype(float)
        h_acc_valid = h_pos_gt > 0
        h_acc[h_acc_valid] = h_tp[h_acc_valid] / h_pos_gt[h_acc_valid]
        h_union = h_pos_gt + h_pos_pred - h_tp
        h_iou_valid = np.logical_and(h_acc_valid, h_union > 0)
        h_iou[h_iou_valid] = h_tp[h_iou_valid] / h_union[h_iou_valid]
        h_macc = np.sum(h_acc[h_acc_valid]) / np.sum(h_acc_valid)
        h_miou = np.sum(h_iou[h_iou_valid]) / np.sum(h_iou_valid)
        h_fiou = np.sum(h_iou[h_iou_valid] * h_class_weights[h_iou_valid])
        h_pacc = np.sum(h_tp) / np.sum(h_pos_gt)
        
        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mAcc"] = 100 * macc
        res["pAcc"] = 100 * pacc
        res["h_mIoU"] = 100 * h_miou
        res["h_fwIoU"] = 100 * h_fiou
        res["h_mAcc"] = 100 * h_macc
        res["h_pAcc"] = 100 * h_pacc
        

        
        # Soft S-measure metrics
        res["Soft-S-measure"] = 100 * np.mean(self._soft_s_measures) if self._soft_s_measures else 0.0
        res["Soft-S-object"] = 100 * np.mean(self._soft_s_object_overall) if self._soft_s_object_overall else 0.0
        res["Soft-S-region"] = 100 * np.mean(self._soft_s_region_overall) if self._soft_s_region_overall else 0.0
        for c_id, class_name in enumerate(self.class_names):
            if c_id in self._class_soft_s_objects and len(self._class_soft_s_objects[c_id]) > 0:
                res[f"Soft-S-object-{class_name}"] = 100 * np.mean(self._class_soft_s_objects[c_id])
            if c_id in self._class_soft_s_regions and len(self._class_soft_s_regions[c_id]) > 0:
                res[f"Soft-S-region-{class_name}"] = 100 * np.mean(self._class_soft_s_regions[c_id])

        # Hard S-measure metrics
        res["Hard-S-measure"] = 100 * np.mean(self._hard_s_measures) if self._hard_s_measures else 0.0
        res["Hard-S-object"] = 100 * np.mean(self._hard_s_object_overall) if self._hard_s_object_overall else 0.0
        res["Hard-S-region"] = 100 * np.mean(self._hard_s_region_overall) if self._hard_s_region_overall else 0.0
        for c_id, class_name in enumerate(self.class_names):
            if c_id in self._class_hard_s_objects and len(self._class_hard_s_objects[c_id]) > 0:
                res[f"Hard-S-object-{class_name}"] = 100 * np.mean(self._class_hard_s_objects[c_id])
            if c_id in self._class_hard_s_regions and len(self._class_hard_s_regions[c_id]) > 0:
                res[f"Hard-S-region-{class_name}"] = 100 * np.mean(self._class_hard_s_regions[c_id])
    
        # --- Write results to text files in a tabular format ---
        if self._output_dir and is_main_process():
            # 1. Write overall metrics to overall_results.txt
            overall_file = os.path.join(self._output_dir, "overall_results.txt")
            main_metrics = {
                "mIoU": res.get("mIoU", 0.0),
                "fwIoU": res.get("fwIoU", 0.0),
                "mAcc": res.get("mAcc", 0.0),
                "pAcc": res.get("pAcc", 0.0),
                "h_mIoU": res.get("h_mIoU", 0.0),
                "h_fwIoU": res.get("h_fwIoU", 0.0),
                "h_mAcc": res.get("h_mAcc", 0.0),
                "h_pAcc": res.get("h_pAcc", 0.0),
                "Soft-S-measure": res.get("Soft-S-measure", 0.0),
                "Soft-S-object": res.get("Soft-S-object", 0.0),
                "Soft-S-region": res.get("Soft-S-region", 0.0),
                "Hard-S-measure": res.get("Hard-S-measure", 0.0),
                "Hard-S-object": res.get("Hard-S-object", 0.0),
                "Hard-S-region": res.get("Hard-S-region", 0.0),
            }
            
            with open(overall_file, 'w') as f:
                max_key_len = max(len(k) for k in main_metrics.keys())
                f.write(f"{'Metric':<{max_key_len}}\tValue\n")
                f.write("-" * (max_key_len + 8) + "\n")
                for key, value in main_metrics.items():
                    f.write(f"{key:<{max_key_len}}\t{value:.4f}\n")

            # 2. Write per-class metrics to per_class_results.txt
            per_class_file = os.path.join(self._output_dir, "per_class_results.txt")
            with open(per_class_file, 'w') as f:
                # Write header
                header = [
                    "Class", "IoU", "h_IoU", 
                    "Soft-S-obj", "Soft-S-reg", "Hard-S-obj", "Hard-S-reg"
                ]
                f.write("\t".join(header) + "\n")
                
                # Write data for each class
                for c_id, class_name in enumerate(self.class_names):
                    # Safely get data using .get(key, default_value)
                    iou_val = 100 * iou[c_id] if iou_valid[c_id] else "N/A"
                    h_iou_val = 100 * h_iou[c_id] if h_iou_valid[c_id] else "N/A"
                          
                    soft_s_obj = res.get(f"Soft-S-object-{class_name}", "N/A")
                    soft_s_reg = res.get(f"Soft-S-region-{class_name}", "N/A")
                    hard_s_obj = res.get(f"Hard-S-object-{class_name}", "N/A")
                    hard_s_reg = res.get(f"Hard-S-region-{class_name}", "N/A")
                    
                    row_data = [
                        class_name,
                        f"{iou_val:.4f}" if isinstance(iou_val, float) else iou_val,
                        f"{h_iou_val:.4f}" if isinstance(h_iou_val, float) else h_iou_val,
                        f"{soft_s_obj:.4f}" if isinstance(soft_s_obj, float) else soft_s_obj,
                        f"{soft_s_reg:.4f}" if isinstance(soft_s_reg, float) else soft_s_reg,
                        f"{hard_s_obj:.4f}" if isinstance(hard_s_obj, float) else hard_s_obj,
                        f"{hard_s_reg:.4f}" if isinstance(hard_s_reg, float) else hard_s_reg,
                    ]
                    f.write("\t".join(row_data) + "\n")
        # --- End of modification ---
                        
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
