# Standard Library imports:
import gc
from typing import Dict

# 1st Party imports:
from .coco import coco_evaluation
from .voc import voc_evaluation
from ssd.data.datasets import COCODataset, VOCDataset, XVIEWCOCODataset, UCBCOCODataset


# class EvaluationMetrics:
#     def __init__(self, dataset, evaluation_result):
#         if isinstance(dataset, VOCDataset):
#             self._parse_pascal_eval_metrics(evaluation_result)
#         elif isinstance(dataset, COCODataset):
#             self._parse_coco_eval_metrics(evaluation_result)
#         elif isinstance(dataset, XVIEWCOCODataset):
#             self._parse_coco_eval_metrics(evaluation_result)

#     def _parse_coco_eval_metrics(self, evaluation_result):
#         self.info = {
#             "AP_IoU=0.50:0.95": evaluation_result.stats[0],
#             "AP_IoU=0.50": evaluation_result.stats[1],
#             "AP_IoU=0.75": evaluation_result.stats[2],
#         }
#         self.eval_stats = evaluation_result.stats
#         del(evaluation_result)
#         gc.collect()

#     def _parse_pascal_eval_metrics(self, evaluation_result):
#         self.info = {"mAP": evaluation_result["map"]}

#     def get_printable_metrics(self):
#         return self.info


def evaluate(dataset, predictions, output_dir, **kwargs) -> Dict[str, float]:
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image. And the index should match the dataset index.
        output_dir: output folder, to save evaluation files or results.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_dir=output_dir, **kwargs
    )
    if isinstance(dataset, VOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, XVIEWCOCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, UCBCOCODataset):
        return coco_evaluation(**args)
    else:
        raise NotImplementedError
