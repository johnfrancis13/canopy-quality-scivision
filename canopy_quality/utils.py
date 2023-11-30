import torch
# from torchmetrics.classification import BinaryJaccardIndex

def custom_replace(tensor, cutpoint):
    res = tensor.clone()
    res[tensor>=cutpoint] = 1
    res[tensor<cutpoint] = 0
    return res

# # Metrics
# device = "cuda" if torch.cuda.is_available() else "cpu"
# def calculate_iou(iou_device=device, pred_tree_mask=[], true_tree_mask =[]):
#     iou_score = BinaryJaccardIndex().to(iou_device)
#     iou_score(torch.squeeze(pred_tree_mask),true_tree_mask.squeeze().type(torch.LongTensor).to(iou_device))

