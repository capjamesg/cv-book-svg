import base64
import concurrent.futures
import csv
import os
from io import BytesIO
import requests

import cv2
import numpy as np
import supervision as sv
from inference.models.utils import get_roboflow_model
from openai import OpenAI

client = OpenAI()

model = get_roboflow_model(model_id="open-shelves/8", api_key=os.getenv("ROBOFLOW_API_KEY"))
smoother = sv.DetectionsSmoother()
tracker = sv.ByteTrack(minimum_consecutive_frames=5)
box_annotator = sv.BoundingBoxAnnotator()
unique_books = []
tracked_books = set()


def process_mask(region, task_id):
    region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

    base64_image = base64.b64encode(
        BytesIO(cv2.imencode(".jpg", region)[1]).read()
    ).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Read the text on the book spine. Only say the book cover title and author if you can find them. Say the book that is most prominent. Return the format [title] | [author], with no punctuation. If you can't find the information, say only NONE. If there are two or more books, return information for both books in the form [title] | [author] [new line] [title] | [author], etc.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    if response.choices[0].message.content == "NONE":
        return None, task_id

    return response.choices[0].message.content.rstrip("Title:"), task_id


def callback(frame: np.ndarray, frame_number: int) -> np.ndarray:
    print(f"Processing frame {frame_number}")

    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    tracked_this_frame = set()

    for idx, detection in enumerate(detections.tracker_id):
        if detection not in tracked_books:
            tracked_books.add(detection)
            tracked_this_frame.add({"tracker_id": detection, "xyxy": detections.xyxy[idx]})

    # order tracked_this_frame by xyxy[0]
    # this allows us to preserve order of books in the frame
    tracked_this_frame = sorted(tracked_this_frame, key=lambda x: x["xyxy"][0])
    
    for idx, tracked_book in enumerate(tracked_this_frame):
        unique_books.append(sv.crop_image(frame, tracked_book["xyxy"]))

    return box_annotator.annotate(frame.copy(), detections=detections)


sv.process_video(source_path="books.mov", target_path="result.mp4", callback=callback)

os.makedirs("books", exist_ok=True)

for file in os.listdir("books"):
    os.remove(os.path.join("books", file))

for idx, track in enumerate(unique_books):
    cv2.imwrite(f"books/{idx}.jpg", track)

ordered_books = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    tasks = [
        executor.submit(process_mask, region, task_id)
        for task_id, region in enumerate(unique_books)
    ]
    books = [task.result() for task in tasks]

    # order by task_id
    books.sort(key=lambda x: x[1])

    for book in books:
        ordered_books.append(book[0])

def process_book_with_google_books(book):
    response = requests.get(
        f"https://www.googleapis.com/books/v1/volumes?q={book}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response = response.json()

    isbn, author, link, description = "NULL", "NULL", "NULL", "NULL"

    try:
        isbn = response["items"][0]["volumeInfo"]["industryIdentifiers"][0]["identifier"]
        if (
            "volumeInfo" in response["items"][0]
            and "authors" in response["items"][0]["volumeInfo"]
        ):
            author = response["items"][0]["volumeInfo"]["authors"][0]
        link = response["items"][0]["volumeInfo"]["infoLink"]
        description = response["items"][0]["volumeInfo"]["description"]
    except:
        pass

    return isbn, author, link, description

saved_books = {}

with open("books.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Title", "Author", "ISBN", "Link", "Description"])

    for book in ordered_books:
        if not book:
            continue

        for b in book.split("\n"):
            title, author = b.strip().split("|")

            if tuple([title, author]) in saved_books:
                continue

            isbn, author, link, description = process_book_with_google_books(title)

            writer.writerow([title, author, isbn, link, description])

            saved_books[tuple([title, author])] = True
