
import os, cv2, numpy as np, torch
def _ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def draw_boxes(image: np.ndarray, boxes: np.ndarray, scores=None, labels=None, color=(0,255,0)) -> np.ndarray:
    img=image.copy()
    for i,b in enumerate(boxes):
        x1,y1,x2,y2=[int(v) for v in b]; cv2.rectangle(img,(x1,y1),(x2,y2),color,2); text=""
        if labels is not None: text += f"cls:{int(labels[i])} "
        if scores is not None: text += f"{float(scores[i]):.2f}"
        if text: cv2.putText(img,text,(x1,max(0,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2,cv2.LINE_AA)
    return img
@torch.no_grad()
def visualize_predictions(model, dataset, device, out_dir: str, num_images: int = 5, score_thr: float = 0.5):
    _ensure_dir(out_dir); model.eval(); count=0
    for i in range(len(dataset)):
        image, target = dataset[i]; img_t=image.to(device).unsqueeze(0); out=model(img_t)[0]
        keep=out["scores"]>=score_thr; boxes=out["boxes"][keep].cpu().numpy(); scores=out["scores"][keep].cpu().numpy(); labels=out["labels"][keep].cpu().numpy()
        img_np=(image.permute(1,2,0).cpu().numpy()*255.0).astype(np.uint8)[:, :, ::-1]
        vis=draw_boxes(img_np, boxes, scores, labels, color=(0,255,0))
        if target["boxes"].numel()>0: vis=draw_boxes(vis, target["boxes"].cpu().numpy(), None, target["labels"].cpu().numpy(), color=(0,0,255))
        out_path=os.path.join(out_dir, f"vis_{i:04d}.jpg"); cv2.imwrite(out_path, vis); count+=1
        if count>=num_images: break
