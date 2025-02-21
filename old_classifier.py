import os
import json
from ultralytics import YOLO

IMG_COUNT = 100
CURRENT_IMG_COUNT = 0

def input_file_list(dir, ext):
    matches = []
    for root, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(ext):
                matches.append(os.path.join(root, filename))
                if len(matches) >= IMG_COUNT:
                    return matches
    return matches

CLIP_ID = "clip_tmp"
INPUT_ROOT = "./downloads"
INPUT_DIR = "./downloads/"+CLIP_ID
INPUT_FILES = input_file_list(INPUT_DIR, ".jpg")
INPUT_FILES = [x.replace("\\", "/") for x in INPUT_FILES]
IMG_COUNT = len(INPUT_FILES)

MODEL = YOLO("yolo11n.pt")

def model_plot_to_image(input_files, segment):
    print(f"predicting {len(input_files)} files")
    results = MODEL.track(input_files)
    print(f"prediciton done. now saving img...")
    tracks = dict()
    for (input_file, result) in zip(input_files, results):
        # output_parent = os.path.dirname(output_file)
        # os.makedirs(output_parent, exist_ok=True)

        # output_file_json = output_file + ".json"
        # print(f"{input_file} ---> {output_file}  + {output_file_json}")
        output_json = json.loads(result.to_json())
        for output_track in output_json:
            if output_track["name"] == "person":
                track_id = str(output_track["track_id"]).zfill(4)
                tracks[track_id] = (tracks.get(track_id) or []) + [{
                    "box": output_track["box"], "img": input_file
                }]
    for track_id in list(tracks.keys()):
        if len(tracks[track_id]) < 20:
            print(f"SKIP TRACK {track_id} LEN {len(tracks[track_id])}")
            del tracks[track_id]
        else:
            print(f"SAVE TRACK {track_id} LEN {len(tracks[track_id])}")
    json_file = f"{INPUT_ROOT}/tracks/{CLIP_ID}/{segment}/tracks.json"
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    print(f"SAVE {json_file}")
    with open(json_file, 'w') as f:
        f.write(json.dumps(tracks, indent=2))

BATCH_SIZE = min(100, IMG_COUNT)

for i in range(0, int(IMG_COUNT/BATCH_SIZE)):
    input = INPUT_FILES[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    model_plot_to_image(input, str(i).zfill(3))
