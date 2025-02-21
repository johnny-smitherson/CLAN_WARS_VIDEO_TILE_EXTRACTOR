import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from tracker2 import preprocess_frame


def sharpen_filter(image, strength=1.0):
    if image is None:
        raise ValueError("Input image cannot be None")
    
    if not 0.0 <= strength <= 5.0:
        raise ValueError("Strength must be between 0.0 and 5.0")
    
    # Create sharpening kernel with strength factor
    kernel = np.array([
        [0, -1 * strength, 0],
        [-1 * strength, 1 + 4 * strength, -1 * strength],
        [0, -1 * strength, 0]
    ])
    
    # Apply filter and clip values to valid range
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)



def run_model(model, im0):
    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True, )#imgsz=im0.shape[:2])

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls = results[0].boxes.cls.cpu().tolist()

        for mask, track_id, cls in zip(masks, track_ids, cls):
            if cls != 0:
                continue
            color = colors(int(track_id), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color, label=str(track_id), txt_color=txt_color)
    return im0


def preview_preprocess():



    model = YOLO("yolo11l-seg.pt")  # segmentation model
    cap = cv2.VideoCapture("downloads/clip_y1.webm")
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        im0_orig = im0.copy()
        im0 = preprocess_frame(im0)
        im0 = sharpen_filter(im0, strength=5.0)
        im0 = cv2.GaussianBlur(im0, (5, 5), 0)
        im0 = sharpen_filter(im0, strength=5.0)
        im0 = cv2.GaussianBlur(im0, (5, 5), 0)

        im0 = run_model(model, im0)
        # track_orig = run_model(model, im0_orig)

        cv2.imshow("original", im0_orig)
        # cv2.imshow("track_orig", track_orig)
        cv2.imshow("track_preprocess", im0)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    preview_preprocess()

