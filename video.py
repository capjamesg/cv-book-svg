import cv2
import supervision as sv
import tqdm
from inference import get_roboflow_model
from PIL import Image
import os

FILE = "books.mov"

unique_books = []
tracked_ids = set()

tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother()

model = get_roboflow_model(model_id="bookshelf-digitizer/1")

for frame in tqdm.tqdm(sv.get_video_frames_generator(source_path=FILE)):
    results = model.infer(frame)
    detections = sv.Detections.from_inference(
        results[0].dict(by_alias=True, exclude_none=True)
    )

    predictions = tracker.update_with_detections(detections)
    predictions = smoother.update_with_detections(predictions)

    for prediction in predictions:
        tracker_id = prediction[4]

        if tracker_id not in tracked_ids:
            tracked_ids.add(tracker_id)
            x0 = prediction[0][0]
            y0 = prediction[0][1]
            x1 = prediction[0][2]
            y1 = prediction[0][3]
            roi = Image.fromarray(frame).crop((x0, y0, x1, y1))
            unique_books.append(roi)

            print(f"Found {len(unique_books)} unique books so far.")

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, detections=predictions
    )

    cv2.imshow("frame", frame)
    cv2.waitKey(1)

if not os.path.exists("books"):
    os.makedirs("books")

for i, book in enumerate(unique_books):
    book.save(f"books/book_{i}.png")
