import argparse
import os
import sys
import numpy as np
import json
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

# Import GroundingDINO dependencies
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip() + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenized = model.tokenizer(caption)
    pred_phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, model.tokenizer) + f"({logit.max().item():.2f})"
        for logit in logits_filt
    ]
    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
    mask_image = mask.reshape(mask.shape[-2], mask.shape[-1], 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    mask_img = torch.zeros(mask_list.shape[-2:])
    json_data = [{'value': 0, 'label': 'background'}]

    for idx, (mask, label, box) in enumerate(zip(mask_list, label_list, box_list)):
        mask_img[mask.cpu().numpy()[0] == True] = idx + 1
        name, logit = label.split('(')
        logit = logit[:-1]
        box_data = {
            'value': idx + 1,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
            'contour': []
        }

        mask_np = (mask.cpu().numpy().astype(np.uint8) * 255).squeeze(0)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            box_data['contour'].append(contour.squeeze(1).tolist())
        json_data.append(box_data)

    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f, indent=4)

    return json_data


def process_image(input_image, text_prompt, output_dir="outputs",
                  config="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                  grounded_checkpoint="groundingdino_swint_ogc.pth",
                  sam_version="vit_b", sam_checkpoint="sam_vit_b_01ec64.pth",
                  box_threshold=0.3, text_threshold=0.25, device="cuda",
                  bert_base_uncased_path=None, use_sam_hq=False, sam_hq_checkpoint=None):
    
    os.makedirs(output_dir, exist_ok=True)
    image_pil, image = load_image(input_image)
    model = load_model(config, grounded_checkpoint, bert_base_uncased_path, device)
    boxes_filt, pred_phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device=device)

    predictor = (sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device) 
                 if use_sam_hq else sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    predictor = SamPredictor(predictor)

    image = cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    H, W = image_pil.size[1], image_pil.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt.cpu(), image.shape[:2]).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords=None, point_labels=None, boxes=transformed_boxes.to(device), multimask_output=False
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.cpu().numpy(), plt.gca(), label)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)

    return save_mask_data(output_dir, masks, boxes_filt, pred_phrases)


if __name__=="__main__":
    # 使用示例
    contour_data = process_image(input_image="latest_frame.jpg", text_prompt="road")
    print(contour_data)
