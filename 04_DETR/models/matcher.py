# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class    # 2024.05.14 @hslee : (Paper says) hyperparameter to balance the importance of the classification error
        self.cost_bbox = cost_bbox      # 2024.05.14 @hslee : (Paper says) hyperparameter to balance the importance of the L1 error of the bounding box coordinates
        self.cost_giou = cost_giou      # 2024.05.14 @hslee : (Paper says) hyperparameter to balance the importance of the giou loss of the bounding box
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 논문 설명에서처럼, hungarian algorithm으로 optimal assignment를 찾고 나서 loss를 구한게 아니라,
        # hungarian algorithm을 위해 cost matrix를 구성할 때, loss값으로 cost matrix를 만들어서 hungarian algorithm을 진행함.
        # 그래서, cost matrix는 모든 matching에 대한 loss값을 갖고 있음

        # We flatten to compute the cost matrices in a batch 
        # + 2024.05.14 @hslee : (Paper says) This makes the class prediction term commensurable to out_bbox
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes], 
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # loss1 : classification loss
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # loss2_1 : bbox loss
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # print(f"cost_bbox.shape: {cost_bbox.shape}")

        # loss2_2 : giou loss
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # final_loss = (hyper_param1 * loss1) + (hyper_param2_1 * loss2_1) + (hyper_param2_2 * loss2_2)
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            # (bs * num_queries, #matchings)
            
        # Final cost matrix = final_loss.view(bs, num_queries, #matchings)
        C = C.view(bs, num_queries, -1).cpu()
            # (bs, num_queries, #matchings)

        sizes = [len(v["boxes"]) for v in targets]
            # [batch1's #matchings, ..., batchN's #matchings]
        
        # linear_sum_assigment : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        # hungarian algorithm을 이용하여 optimal assignment를 찾음
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        '''
            sizes: [1, 11, 1, 8]
            indices: [(array([58]), array([0])), \
                    (array([ 7, 14, 28, 36, 46, 47, 50, 56, 70, 71, 87]), array([ 9,  5,  7,  2,  8,  4,  3, 10,  6,  0,  1])), \
                    (array([92]), array([0])), \
                    (array([27, 31, 46, 51, 55, 65, 66, 82]), array([6, 7, 5, 1, 3, 4, 2, 0]) \
                    )]
        
            4개 batch에 대한 각각의 target box에 대한 prediction box의 index를 반환함.
                1st batch,  1개 matching : array([58]), array([0])                  : 58번쨰 prediction box와 0번쨰 target box가 matching됨
                2nd batch, 11개 matching : array([ 7,..., 87]), array([ 9, ..., 1]) : [7, ..., 87]번쨰 prediction box와 [ 9, ..., 1]번쨰 target box가 각각 matching됨
                3th batch,  1개 matching : array([92]), array([0])                  : 92번쨰 prediction box와 0번쨰 target box가 matching됨
                4th batch,  8개 matching : array([ 4, ..., 91]), array([3, ..., 2]) : [4, ..., 91]번쨰 prediction box와 [ 3, ..., 2]번쨰 target box가 각각 matching됨
        '''
        
        res = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        
        return res


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
