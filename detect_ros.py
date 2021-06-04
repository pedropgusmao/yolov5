import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
    save_one_box,
)
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


@torch.no_grad()
def detect(opt):
    source, weights, view_img, imgsz = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.img_size,
    )

    # Load model
    model = attempt_load(weights)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = (
        model.module.names if hasattr(model, "module") else model.names
    )  # get class names

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            opt.classes,
            opt.agnostic_nms,
            max_det=opt.max_det,
        )

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s.copy()

            imc = im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f"{names[c]} {conf:.2f}"
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors(c, True),
                            line_thickness=opt.line_thickness,
                        )

            # Stream results
            if view_img:
                cv2.imshow("Prediction", im0)
                cv2.waitKey(0)  # 1 millisecond


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov5s.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--source", type=str, default="data/images", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=1000,
        help="maximum number of detections per image",
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    opt = parser.parse_args()
    check_requirements(exclude=("tensorboard", "pycocotools", "thop"))

    detect(opt=opt)


if __name__ == "__main__":
    main()
