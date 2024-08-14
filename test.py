# import cv2
# import numpy as np
# import torch
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.config import get_cfg
# from detectron2.engine import default_setup
# from detectron2.modeling import build_model
# from detectron2.projects.deeplab import add_deeplab_config

# from maskdino.config import add_maskformer2_config

# # #####################################3
# WEIGHTS_PATH = "./model_final_doclay_swindocseg.pth"
# CONFIG_PATH = "./config_doclay.yaml"
# # Class names in the proper order
# # PRIMA config and weights:
# # CLASSES = np.asarray(
# #     [
# #         "Background",
# #         "TextRegion",
# #         "ImageRegion",
# #         "TableRegion",
# #         "MathsRegion",
# #         "SeparatorRegion",
# #         "OtherRegion",
# #     ]
# # )
# # OR
# # DocLayNet config and weights:
# CLASSES = np.asarray(
#     [
#         "Caption",
#         "Footnote",
#         "Formula",
#         "List-item",
#         "Page-footer",
#         "Page-header",
#         "Picture",
#         "Section-header",
#         "Table",
#         "Text",
#         "Title",
#     ]
# )
# # #####################################3

# # Color for each class
# COLORS = [
#     (255, 0, 0),
#     (0, 255, 0),
#     (0, 0, 255),
#     (255, 255, 0),
#     (0, 255, 255),
#     (255, 0, 255),
#     (255, 255, 255),
#     (127, 0, 0),
#     (0, 127, 0),
#     (0, 0, 127),
#     (127, 127, 127),
# ]


# def setup(args):
#     """
#     create configs and perform basic setups.
#     """

#     cfg = get_cfg()
#     # for poly lr schedule
#     add_deeplab_config(cfg)
#     add_maskformer2_config(cfg)
#     cfg.merge_from_file(args["config_file"])
#     cfg.merge_from_list(args["opts"])
#     cfg.freeze()
#     default_setup(cfg, args)
#     # shutup detectron2 completely
#     # logger = setup_logger(
#     #     output=cfg.output_dir,
#     #     distributed_rank=comm.get_rank(),
#     #     name="maskdino",
#     #     configure_stdout=false,
#     # )
#     # logger.setlevel("error")
#     return cfg


# def draw_box(img, box, color, label, score):
#     # box: [x0,y0,x1,y1]
#     box = [round(x) for x in box]
#     color = COLORS[color]

#     # draw a bounding box rectangle and label on the image
#     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
#     text = "{}: {:.4f}".format(CLASSES[label], score)
#     cv2.putText(
#         img,
#         text,
#         (box[0], box[1] - 5),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         color,
#         2,
#     )

#     return img


# # Load model and weights
# cfg = setup({"config_file": CONFIG_PATH, "opts": []})
# model = build_model(cfg)  # returns a torch.nn.module
# DetectionCheckpointer(model).load(WEIGHTS_PATH)

# img1 = cv2.imread("/home/akash/ws/dataset/datasets/images/val/4322.png")
# img2 = cv2.imread("/home/akash/ws/dataset/datasets/images/val/4322.png")

# def preprocess_image(img: np.ndarray):
#     # resize for faster inference and RAM saving
#     img = cv2.resize(img, (img.shape[1], img.shape[0]))
#     # img = cv2.resize(img, (1024, 1024))
#     # bgr to rgb
#     img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # hwc to chw format
#     img_ = torch.tensor(img_).permute(2, 0, 1)
#     # float32 format
#     img_ = img_.float().cuda()
#     return img_


# # inference
# model.eval()
# with torch.no_grad():
#     for j, img in enumerate([img1, img2]):
#         img_ = preprocess_image(img)
#         outputs = model(
#             [{"image": img_, "height": img_.shape[1], "width": img_.shape[2]}]
#         )

#         # Draw boxes on image
#         for output in outputs:
#             pred_classes = output["instances"].pred_classes.cpu().numpy()
#             boxes = output["instances"].pred_boxes.tensor.cpu().numpy()
#             scores = output["instances"].scores.cpu().numpy()

#             # filter out low confidence boxes
#             boxes = boxes[scores > 0.4]
#             pred_classes = pred_classes[scores > 0.4]
#             final_scores = scores[scores>0.4]

#             # draw boxes on image
#             for i, box in enumerate(boxes):
#                 # box *= 2
#                 img = draw_box(img, box, pred_classes[i], pred_classes[i], final_scores[i])

#         # save image
#         cv2.imwrite(f"example_segmented{j}.png", img)



import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
from maskdino.config import add_maskformer2_config

class DocumentLayoutSegmentation:
    def __init__(self, config_path, weights_path, classes, colors, conf_threshold=0.4):
        self.config_path = config_path
        self.weights_path = weights_path
        self.classes = classes
        self.colors = colors
        self.conf_threshold = conf_threshold
        self.cfg = self.setup({"config_file": self.config_path, "opts": []})
        self.model = self.load_model()

    def setup(self, args):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args["config_file"])
        cfg.merge_from_list(args["opts"])
        cfg.freeze()
        default_setup(cfg, args)
        return cfg

    def load_model(self):
        model = build_model(self.cfg)
        DetectionCheckpointer(model).load(self.weights_path)
        model.eval()
        return model

    def preprocess_image(self, img: np.ndarray):
        img = cv2.resize(img, (img.shape[1], img.shape[0]))
        img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_ = torch.tensor(img_).permute(2, 0, 1).float().cuda()
        return img_

    def draw_box(self, img, box, color, label, score):
        box = [round(x) for x in box]
        color = self.colors[color]
        text = "{}: {:.4f}".format(self.classes[label], score)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(
            img,
            text,
            (box[0], box[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        return img

    def segment_image(self, img_path, save_path):
        img = cv2.imread(img_path) if img_path is str else img_path
        img_ = self.preprocess_image(img)

        with torch.no_grad():
            outputs = self.model([{"image": img_, "height": img_.shape[1], "width": img_.shape[2]}])

            for output in outputs:
                pred_classes = output["instances"].pred_classes.cpu().numpy()
                boxes = output["instances"].pred_boxes.tensor.cpu().numpy()
                scores = output["instances"].scores.cpu().numpy()

                # filter out low confidence boxes
                mask = scores > self.conf_threshold
                boxes = boxes[mask]
                pred_classes = pred_classes[mask]
                final_scores = scores[mask]

                # draw boxes on image
                for i, box in enumerate(boxes):
                    img = self.draw_box(img, box, pred_classes[i], pred_classes[i], final_scores[i])

        cv2.imwrite(save_path, img)
        return pred_classes,boxes, final_scores

# Example usage:
if __name__ == "__main__":
    CLASSES = np.asarray(
        [
            "Caption",
            "Footnote",
            "Formula",
            "List-item",
            "Page-footer",
            "Page-header",
            "Picture",
            "Section-header",
            "Table",
            "Text",
            "Title",
        ]
    )

    COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (127, 0, 0),
        (0, 127, 0),
        (0, 0, 127),
        (127, 127, 127),
    ]

    # config_path = "./config_doclay.yaml"
    # weights_path = "./model_final_doclay_swindocseg.pth"

    # segmenter = DocumentLayoutSegmentation(config_path, weights_path, CLASSES, COLORS)
    # segmenter.segment_image("/home/akash/ws/dataset/datasets/images/val/4322.png", "example_segmented.png")
