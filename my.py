import mimetypes
import os
import tempfile
from argparse import ArgumentParser

import json_tricks as json
import mmcv
import mmengine
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
import cv2
import matplotlib.pyplot as plt

# try:
#     from mmdet.apis import inference_detector, init_detector
#     has_mmdet = True
# except (ImportError, ModuleNotFoundError):
#     has_mmdet = False

# def getPersonID(bbox, sample_bbox):
#     temp = 100000
#     idx = 0
#     for key in sample_bbox:
#         distance = (bbox[0] - sample_bbox[key][0]) ** 2 + (bbox[1] - sample_bbox[key][1]) ** 2
#         if distance < temp:
#             idx = key
#             temp = distance
#     return idx


# def process_one_image(args, img_path, detector, pose_estimator, visualizer,
#                       show_interval, sample_bbox):
#     det_result = inference_detector(detector, img_path)
#     pred_instance = det_result.pred_instances.cpu().numpy()
#     bboxes = np.concatenate(
#         (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
#     bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
#                                    pred_instance.scores > args.bbox_thr)]
#     confident = bboxes[nms(bboxes, args.nms_thr), 4] * 100
#     # print('!!!', confident)
    
#     bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
#     pose_results = inference_topdown(pose_estimator, img_path, bboxes)
#     data_samples = merge_data_samples(pose_results)

#     img = mmcv.imread(img_path, channel_order='rgb')
#     for i, bbox in enumerate(bboxes):
#         idx = getPersonID(bbox, sample_bbox)
#         cv2.putText(img,f"Person{idx} {confident[i]:.2f}%", (int(bbox[0]),int(bbox[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

#     out_file = None
#     if args.output_root:
#         out_file = f'{args.output_root}/{os.path.basename(img_path)}'

#     visualizer.add_datasample(
#         'result',
#         img,
#         data_sample=data_samples,
#         draw_gt=False,
#         draw_heatmap=args.draw_heatmap,
#         draw_bbox=1,
#         show_kpt_idx=args.show_kpt_idx,
#         skeleton_style=args.skeleton_style,
#         show=args.show,
#         wait_time=show_interval,
#         out_file=out_file,
#         kpt_thr=args.kpt_thr)
#     return data_samples.get('pred_instances', None)


# def process_first_image(args, img_path, detector):
#     det_result = inference_detector(detector, img_path)
#     pred_instance = det_result.pred_instances.cpu().numpy()
#     bboxes = np.concatenate(
#         (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
#     bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
#                                    pred_instance.scores > args.bbox_thr)]
#     bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
#     sample = {}
#     for i, bbox in enumerate(bboxes):
#         sample[i] = [bbox[0], bbox[1]]
#     return sample

# def main():
#     parser = ArgumentParser()
#     parser.add_argument('--det-config', default='demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py', help='Config file for detection')
#     parser.add_argument('--det-checkpoint', default='faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', help='Checkpoint file for detection')
#     parser.add_argument('--pose-config', default='configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py', help='Config file for pose')
#     parser.add_argument('--pose-checkpoint', default='hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth', help='Checkpoint file for pose')

    
#     # parser.add_argument('--det-config', default='demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py', help='Config file for detection')
#     # parser.add_argument('--det-checkpoint', default='faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', help='Checkpoint file for detection')
#     # parser.add_argument('--pose-config', default='configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py', help='Config file for pose')
#     # parser.add_argument('--pose-checkpoint', default='second.pth', help='Checkpoint file for pose')

#     parser.add_argument(
#         '--input', type=str, default='', help='Image/Video file')
#     parser.add_argument(
#         '--show',
#         action='store_true',
#         default=True,
#         help='whether to show img')
#     parser.add_argument(
#         '--output-root',
#         type=str,
#         default='results',
#         help='root of the output img file. '
#         'Default not saving the visualization images.')
#     parser.add_argument(
#         '--save-predictions',
#         action='store_true',
#         default=False,
#         help='whether to save predicted results')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--det-cat-id',
#         type=int,
#         default=0,
#         help='Category id for bounding box detection model')
#     parser.add_argument(
#         '--bbox-thr',
#         type=float,
#         default=0.5,
#         help='Bounding box score threshold')
#     parser.add_argument(
#         '--nms-thr',
#         type=float,
#         default=0.2,
#         help='IoU threshold for bounding box NMS')
#     parser.add_argument(
#         '--kpt-thr',
#         type=float,
#         default=0.3,
#         help='Visualizing keypoint thresholds')
#     parser.add_argument(
#         '--draw-heatmap',
#         action='store_true',
#         default=False,
#         help='Draw heatmap predicted by the model')
#     parser.add_argument(
#         '--show-kpt-idx',
#         action='store_true',
#         default=False,
#         help='Whether to show the index of keypoints')
#     parser.add_argument(
#         '--skeleton-style',
#         default='mmpose',
#         type=str,
#         choices=['mmpose', 'openpose'],
#         help='Skeleton style selection')
#     parser.add_argument(
#         '--radius',
#         type=int,
#         default=2,
#         help='Keypoint radius for visualization')
#     parser.add_argument(
#         '--thickness',
#         type=int,
#         default=1,
#         help='Link thickness for visualization')
#     parser.add_argument(
#         '--alpha', type=float, default=0.8, help='The transparency of bboxes')
#     parser.add_argument(
#         '--draw-bbox', action='store_true', help='Draw bboxes of instances')

#     assert has_mmdet, 'Please install mmdet to run the demo.'

#     args = parser.parse_args()

#     assert args.show or (args.output_root != '')
#     assert args.input != ''
#     assert args.det_config is not None
#     assert args.det_checkpoint is not None
#     if args.output_root:
#         mmengine.mkdir_or_exist(args.output_root)
#     if args.save_predictions:
#         assert args.output_root != ''
#         args.pred_save_path = f'{args.output_root}/results_' \
#             f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

#     detector = init_detector(
#         args.det_config, args.det_checkpoint, device=args.device)
#     detector.cfg = adapt_mmdet_pipeline(detector.cfg)

#     pose_estimator = init_pose_estimator(
#         args.pose_config,
#         args.pose_checkpoint,
#         device=args.device,
#         cfg_options=dict(
#             model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

#     pose_estimator.cfg.visualizer.radius = args.radius
#     pose_estimator.cfg.visualizer.alpha = args.alpha
#     pose_estimator.cfg.visualizer.line_width = args.thickness

#     visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
#     visualizer.set_dataset_meta(
#         pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

#     input_type = mimetypes.guess_type(args.input)[0].split('/')[0]
#     if input_type == 'image':
#         pred_instances = process_one_image(
#             args,
#             args.input,
#             detector,
#             pose_estimator,
#             visualizer,
#             show_interval=0)

#     elif input_type == 'video':
#         tmp_folder = tempfile.TemporaryDirectory()
#         video = mmcv.VideoReader(args.input)
#         progressbar = mmengine.ProgressBar(len(video))
#         video.cvt2frames(tmp_folder.name, show_progress=False)
#         output_root = args.output_root
#         args.output_root = tmp_folder.name
#         sample_bbox = {}
#         for frame_id, img_fname in enumerate(os.listdir(tmp_folder.name)):
#             if frame_id == 1:
#                 sample_bbox = process_first_image(
#                     args,
#                     f'{tmp_folder.name}/{img_fname}',
#                     detector)
#                 break
#         for frame_id, img_fname in enumerate(os.listdir(tmp_folder.name)):
#             pred_instances = process_one_image(
#                 args,
#                 f'{tmp_folder.name}/{img_fname}',
#                 detector,
#                 pose_estimator,
#                 visualizer,
#                 show_interval=0.01,
#                 sample_bbox=sample_bbox)

#             progressbar.update()
#         if output_root:
#             mmcv.frames2video(
#                 tmp_folder.name,
#                 f'{output_root}/{os.path.basename(args.input)}',
#                 fps=video.fps,
#                 fourcc='mp4v',
#                 show_progress=False)
#         tmp_folder.cleanup()

#     else:
#         args.save_predictions = False
#         raise ValueError(
#             f'file {os.path.basename(args.input)} has invalid format.')
# if __name__ == '__main__':
#     main()
import cv2
from matplotlib import pyplot as plt

parser = ArgumentParser()
parser.add_argument('--input', type=str, default='', help='Image/Video file')
args = parser.parse_args()
cap = cv2.VideoCapture(args.input)
ret, frame = cap.read()
plt.axis("off")
while ret:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    plt.imshow(frame)
    plt.pause(0.01)
plt.show()
    


