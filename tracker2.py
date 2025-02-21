import cv2
import time
import numpy as np
import os
import shutil
import json
import subprocess
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


import torch
assert torch.cuda.is_available()


MARGIN = 12
MIN_SCALE=640

def crop_box_from_poly(box_poly, shape):
    x_axis = list(b[0] for b in box_poly)
    y_axis = list(b[1] for b in box_poly)

    x1 = int(min(x_axis)-MARGIN)
    x2 = int(max(x_axis)+MARGIN)
    y1 = int(min(y_axis)-MARGIN)
    y2 = int(max(y_axis)+MARGIN)

    if x2 - x1 < MARGIN:
        return None
    if y2 - y1 < MARGIN:
        return None

    # Skip if crop box is too close to edges

    if (x1 <= MARGIN or 
        y1 <= MARGIN or 
        x2 >= shape[1] - MARGIN or 
        y2 >= shape[0] - MARGIN):
        return None

    x1 = max(x1, 0)
    x2 = min(x2, shape[1])
    y1 = max(y1, 0)
    y2 = min(y2, shape[0])
    return (x1, x2, y1, y2)

def apply_crop_box(crop_box, image):
    x1, x2, y1, y2 = crop_box
    return image[y1:y2+1, x1:x2+1]





# def gamma_correction(image, gamma=1.0):
#     inv_gamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
#     return cv2.LUT(image, table)

# def normalize_image(image):
#     return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def apply_gaussian_blur(image, kernel=(5,5)):
    return cv2.GaussianBlur(image, kernel, 0)


def contrast_stretch(image):
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
    y_channel_stretched = cv2.normalize(y_channel, None, 0, 255, cv2.NORM_MINMAX)
    contrast_stretched_ycrcb = cv2.merge([y_channel_stretched, cr_channel, cb_channel])
    contrast_stretched_image = cv2.cvtColor(contrast_stretched_ycrcb, cv2.COLOR_YCrCb2BGR)
    return contrast_stretched_image

def apply_alpha_mask(image, mask):
    # Convert image to BGRA
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Set alpha channel proportional to mask value
    image[:, :, 3] = mask
    return image


def normalize_alpha_mask(alpha):
    # Normalize mask values to 0-255 range for alpha channel
    alpha = alpha.astype(np.float32) / 255.0
    # increase big alpha values
    alpha = np.sqrt(alpha)
    # normalize alpha values to 0-1 range
    alpha = alpha / (alpha.max()+0.00001)

    alpha = (alpha * 255).astype(np.uint8)
    # remove low-end noise
    alpha[alpha < 4] = 0
    # remove high-end noise
    alpha[alpha > 240] = 255
    return alpha


def soften_alpha_mask(alpha):
    b_mask_blur = alpha.copy()
    b_blur_size = 1+min(alpha.shape[0], alpha.shape[1]) // 20
    if b_blur_size % 2 == 0: b_blur_size += 1
    b_blur_size = (b_blur_size,b_blur_size)
    erode_kern = cv2.getGaussianKernel(3, -1) @ cv2.getGaussianKernel(3, -1).T
    for _x in range(1,4):
        # b_mask_blur = (apply_gaussian_blur(b_mask_blur, b_blur_size) + alpha) / 2
        # Set outermost pixels to zero at start of each iteration
        b_mask_blur[0, :] = 0  # Top row
        b_mask_blur[-1, :] = 0  # Bottom row
        b_mask_blur[:, 0] = 0  # Left column
        b_mask_blur[:, -1] = 0  # Right column
        
        b_mask_blur = (apply_gaussian_blur(b_mask_blur, b_blur_size) + b_mask_blur) / 2
        b_mask_blur = apply_gaussian_blur(b_mask_blur, b_blur_size)
        b_mask_blur = cv2.erode(b_mask_blur, erode_kern, iterations=1)
        b_mask_blur = cv2.dilate(b_mask_blur, erode_kern, iterations=1)
        b_mask_blur = (b_mask_blur + normalize_alpha_mask(b_mask_blur)) / 2
        b_mask_blur = cv2.erode(b_mask_blur, erode_kern, iterations=1)
    return normalize_alpha_mask(b_mask_blur)


def save_frame_crop(video_id, track_id, frame_id, crop, bbox_info):
    track_id = str(track_id).zfill(6)
    frame_id = str(frame_id).zfill(6)
    bbox_info["frame_id"] = frame_id
    img_path = f"downloads/tracks/{video_id}/{track_id}/{frame_id}.png"
    info_path = f"downloads/tracks/{video_id}/{track_id}/000000.ndjson"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    # print("saving crop to ", img_path)
    cv2.imwrite(img_path, crop)
    with open(info_path, "a") as f:
        json.dump(bbox_info, f)
        f.write("\n")

def delete_old_tracks_data(video_id):
    os.makedirs(f"downloads/tracks/{video_id}", exist_ok=True)
    shutil.rmtree(f"downloads/tracks/{video_id}")

    os.makedirs(f"downloads/tracks/{video_id}", exist_ok=True)

def resize_output(image, size):
    # First resize maintaining aspect ratio
    h, w = image.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    
    # Create a transparent square canvas
    result = np.zeros((size, size, 4), dtype=np.uint8)
    
    # Calculate padding to center the image
    h_pad = (size - resized.shape[0]) // 2
    w_pad = (size - resized.shape[1]) // 2
    
    # Place the resized image in the center of the canvas
    result[h_pad:h_pad+resized.shape[0], w_pad:w_pad+resized.shape[1]] = resized
    
    return result

def work_video(video_path, video_id):
    if os.path.exists(f"downloads/tracks/{video_id}"):
        print(f"Skipping {video_id}, folder exists")
        return
    model = YOLO("yolo11x-seg.pt")  # segmentation model
    model.to("cuda")
    model.classes = [0, 2]  # detect only persons and cars
    try:
        do_work_model(model, video_path, video_id)
    except Exception as e:
        print(f"Error processing {video_id}: {e}")

def preprocess_frame(im0_orig):
    im0_orig = contrast_stretch(im0_orig)
    im0_orig = cv2.copyMakeBorder(im0_orig, MARGIN//2, MARGIN//2, MARGIN//2, MARGIN//2, cv2.BORDER_CONSTANT, value=(255,255,255))

    # resize img to ensure minimum dimensions of 2000x2000
    h, w = im0_orig.shape[:2]
    # Calculate initial scale
    scale = max(MIN_SCALE/w, MIN_SCALE/h)
    # Calculate target dimensions
    target_w = int(w * scale)
    target_h = int(h * scale)
    # Round up to nearest multiple of 16
    target_w = ((target_w + 31) // 32) * 32
    target_h = ((target_h + 31) // 32) * 32
    # Resize to the adjusted dimensions
    im0_orig = cv2.resize(im0_orig, (target_w, target_h))
    return im0_orig

def do_work_model(model, video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    w, h, fps, frame_count = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    t0 = time.time()

    while True:
        ret, im0_orig = cap.read()
        output_count = 0
        frame_id += 1
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break 
        
        im0_orig = preprocess_frame(im0_orig)

        results = model.track(im0_orig, persist=True, device="cuda:0", imgsz=im0_orig.shape[:2], verbose=False)

        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for i, (mask, track_id, cls) in enumerate(zip(masks, track_ids, classes)):
                if cls != 0:
                    continue
                
                crop_box = crop_box_from_poly(mask, im0_orig.shape)
                if crop_box is None:  # Skip if crop box is too close to edges
                    continue
                # compute bbox center
                bbox_center = (crop_box[0] + crop_box[1]) / 2, (crop_box[2] + crop_box[3]) / 2
                img_size_avg = (im0_orig.shape[0] + im0_orig.shape[1]) / 2
                bbox_width = (crop_box[1] - crop_box[0])/img_size_avg
                bbox_height = (crop_box[3] - crop_box[2])/img_size_avg
                bbox_aspect = bbox_height / bbox_width
                bbox_size = bbox_width * bbox_height
                if bbox_aspect < 0.9 or bbox_aspect > 2.7:
                    continue
                if bbox_height < 0.02 or bbox_width < 0.01:
                    continue
                if bbox_size < 0.005:
                    continue

                bbox_info = {
                    "center_x": round(bbox_center[0] / img_size_avg, 4),
                    "center_y": round((im0_orig.shape[0] - bbox_center[1]) / img_size_avg, 4),
                    "width": round(bbox_width, 4),
                    "height": round(bbox_height, 4),
                    "aspect": round(bbox_aspect, 4),
                    "area_rel": round(bbox_size, 5)
                }

                # Create binary mask
                alpha_mask = np.zeros(im0_orig.shape[:2], np.uint8)
                contour = mask.astype(np.int32)
                contour = contour.reshape(-1, 1, 2)
                _ = cv2.drawContours(alpha_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                
                crop_orig = apply_crop_box(crop_box, im0_orig)
                alpha_mask = apply_crop_box(crop_box, alpha_mask)
                alpha_mask = soften_alpha_mask(alpha_mask)

                transparent_img = apply_alpha_mask(crop_orig, alpha_mask)
                
                sz = max(transparent_img.shape[0], transparent_img.shape[1])
                transparent_img = resize_output(transparent_img, sz)
                save_frame_crop(video_id,track_id, frame_id, transparent_img, bbox_info)
                output_count += 1

        done_percent = int(frame_id / frame_count * 100)
        t1 = time.time()
        eta = (t1-t0) * (frame_count-frame_id) / frame_id
        eta = int(eta / 60)

        print(f"{frame_id:06d}/{frame_count:06d} DONE {done_percent:.1f}% ETA {eta} min / {output_count} new crops")
    cap.release()
    print(f"Video processing completed: {video_id}.")

def delete_short_tracks():
    for video_id in os.listdir("downloads/tracks"):
        for track_id in os.listdir(f"downloads/tracks/{video_id}"):
            track_length = len(os.listdir(f"downloads/tracks/{video_id}/{track_id}"))
            if track_length < 50:
                print(f"Deleting short ({track_length} frames) track {track_id} in video {video_id}")
                shutil.rmtree(f"downloads/tracks/{video_id}/{track_id}")
        
def export_gifs():
    commands = []
    for video_id in os.listdir("downloads/tracks"):
        for track_id in os.listdir(f"downloads/tracks/{video_id}"):
            gif_path = f"downloads/track_gifs/{video_id}/{track_id}.gif"
            if os.path.exists(gif_path):
                print(f"Skipping {gif_path}, file exists")
                continue
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            frames = []
            for frame_id in os.listdir(f"downloads/tracks/{video_id}/{track_id}"):
                if not frame_id.endswith(".png"):

                    continue
                frame_path = f"downloads/tracks/{video_id}/{track_id}/{frame_id}"
                frames.append(frame_path)
            frames.sort()
            print(f"Exporting {gif_path}")
            commands.append([
                "magick", "convert",  
                *frames,
                '-background', 'White', 
                '-alpha', 'remove',
                "-delay", "2", 
                '-dispose', '3',
                '-resize', '128x128', 
                "-loop", "0", 
                gif_path
            ])
    commands = json.dumps(commands)
    commands = json.loads(commands)
    from concurrent.futures import ThreadPoolExecutor
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            p2 = pool.map(run_command, commands)
        for p in p2:
            print("done", p)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pool.terminate()
    else:
        pool.close()
    pool.join()


def run_command(command):
    try:
        subprocess.check_call(command)
        return len(command)
    except Exception as e:
        print(f"Error running command: {e}")
        return 0



def main():
    work_video("downloads/clip2.webm", "clip_2")
    work_video("downloads/clip1.mkv", "clip_1")
    work_video("downloads/clip3.webm", "clip_3")
    work_video("downloads/clip4.webm", "clip_4")
    work_video("downloads/clip5.webm", "clip_5")

    work_video("downloads/clip6.webm", "clip_6")
    work_video("downloads/clip7.mp4", "clip_7")
    work_video("downloads/clip8.mp4", "clip_8")
    work_video("downloads/clip9.webm", "clip_9")

    delete_short_tracks()
    export_gifs()


if __name__ == "__main__":
    main()

