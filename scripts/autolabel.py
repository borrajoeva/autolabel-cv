import os
import sys
import cv2
import torch
import logging
from PIL import Image
from torchvision.ops import nms
from typing import List

sys.path.append(os.path.join(os.getcwd()))

from GroundingDINO.groundingdino.util.inference import load_model, predict
import GroundingDINO.groundingdino.datasets.transforms as T


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


class AutoLabellingObjectDetect:
    def __init__(self):

        self.ontology = {
            0: {"name": "vidrio", "prompt": "glass object"},
            1: {"name": "papel", "prompt": "paper object"},
            2: {"name": "metal", "prompt": "metal object"},
            3: {"name": "plastico", "prompt": "plastic object"}
        }

        self.box_threshold = 0.2
        self.text_threshold = 0.15

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_dir = "dataset/images"
        self.output_img_dir = "dataset/tagged_images/images_annotated"
        self.output_label_dir = "dataset/tagged_images/labels"

        os.makedirs(self.output_img_dir, exist_ok=True)
        os.makedirs(self.output_label_dir, exist_ok=True)

        self.model = self._load_model()

        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        return load_model(
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "models/groundingdino_swint_ogc.pth",
            device=self.device
        )

    def get_images(self):
        return [
            os.path.join(self.input_dir, f)
            for f in os.listdir(self.input_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    # cxcywh -> xyxy (normalizado)
    def cxcywh_to_xyxy(self, box):
        cx, cy, w, h = box
        return [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2
        ]

    def process(self):

        images = self.get_images()

        for img_path in images:

            img_name = os.path.basename(img_path)

            image_cv = cv2.imread(img_path)
            h, w, _ = image_cv.shape

            image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            image_tensor, _ = self.transform(image_pil, None)
            image_tensor = image_tensor.to(self.device)

            all_boxes = []
            all_scores = []
            all_classes = []
            all_names = []

            # -------------------------
            # DETECCIÓN POR CLASE
            # -------------------------
            for class_id, info in self.ontology.items():

                boxes, logits, _ = predict(
                    model=self.model,
                    image=image_tensor,
                    caption=info["prompt"],
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device
                )

                if len(boxes) == 0:
                    continue

                for i in range(len(boxes)):

                    cxcywh = boxes[i].tolist()
                    xyxy = self.cxcywh_to_xyxy(cxcywh)

                    # pasar a píxeles
                    x1 = xyxy[0] * w
                    y1 = xyxy[1] * h
                    x2 = xyxy[2] * w
                    y2 = xyxy[3] * h

                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(float(logits[i]))
                    all_classes.append(class_id)
                    all_names.append(info["name"])

            if len(all_boxes) == 0:
                print("sin detecciones:", img_name)
                continue

            # -------------------------
            # NMS CORRECTO
            # -------------------------
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(all_scores)

            keep = nms(boxes_tensor, scores_tensor, 0.15)

            boxes_tensor = boxes_tensor[keep]
            all_classes = [all_classes[i] for i in keep]
            all_names = [all_names[i] for i in keep]

            # -------------------------
            # DRAW
            # -------------------------


            colors = {
                0: (255, 80, 80),   # vidrio -> rojo suave
                1: (80, 255, 80),   # papel -> verde
                2: (80, 80, 255),   # metal -> azul
                3: (0, 255, 255)    # plastico -> amarillo
            }

            for box, cls_id, name in zip(boxes_tensor, all_classes, all_names):

                x1, y1, x2, y2 = map(int, box.tolist())

                color = colors.get(cls_id, (0, 255, 0))

                # -------------------------
                # bounding box
                # -------------------------
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 3)

                # -------------------------
                # LABEL (grande + fondo)
                # -------------------------
                label = name.upper()

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                thickness = 2

                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # posición del label (evitar que se salga arriba)
                label_y1 = max(y1 - th - 10, 0)
                label_y2 = y1

                # si se sale por arriba, lo ponemos dentro del bbox
                if y1 - th - 10 < 0:
                    label_y1 = y1
                    label_y2 = y1 + th + 10

                # fondo del texto
                cv2.rectangle(
                    image_cv,
                    (x1, label_y1),
                    (x1 + tw + 10, label_y2),
                    color,
                    -1
                )

                # texto
                cv2.putText(
                    image_cv,
                    label,
                    (x1 + 5, label_y2 - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA
                )

                # -------------------------
                # SAVE
                # -------------------------
                cv2.imwrite(os.path.join(self.output_img_dir, img_name), image_cv)

                txt_path = os.path.join(self.output_label_dir, img_name.replace(".jpg", ".txt"))

                with open(txt_path, "w") as f:
                    for box, cls_id in zip(boxes_tensor, all_classes):
                        x1, y1, x2, y2 = box.tolist()

                        xc = ((x1 + x2) / 2) / w
                        yc = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h

                        f.write(f"{cls_id} {xc} {yc} {bw} {bh}\n")

                print("ok:", img_name)


if __name__ == "__main__":
    app = AutoLabellingObjectDetect()
    app.process()