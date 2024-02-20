import base64
import json
import optparse

import cv2
import numpy as np
import requests
import supervision as sv
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from openai import OpenAI
from io import BytesIO
import concurrent.futures

parser = optparse.OptionParser()

parser.add_option("-i", "--image", dest="image", help="Image to process")
parser.add_option("-o", "--output", dest="output", help="Output file")

args, args2 = parser.parse_args()

image = cv2.imread(args.image)

base_model = GroundedSAM(ontology=CaptionOntology({"book spine": "book spine"}), box_threshold=0.1)

results = base_model.predict(image)

client = OpenAI()

masks_isolated = []

masks_to_xyxys = sv.mask_to_xyxy(masks=results.mask)

polygons = [sv.mask_to_polygons(mask) for mask in results.mask]

for mask in results.mask:
    masked_region = np.zeros_like(image)
    masked_region[mask] = image[mask]
    masks_isolated.append(masked_region)

books = []
links = []

def process_mask(region, task_id):
    region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

    base64_image = base64.b64encode(BytesIO(cv2.imencode(".jpg", region)[1]).read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Read the text on the book spine. Only say the book cover title and author if you can find them. Say the book that is most prominent. Return the format [title] [author], with no punctuation.",
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

    return response.choices[0].message.content.rstrip("Title:").replace("\n", " ")

with concurrent.futures.ThreadPoolExecutor() as executor:
    tasks = [executor.submit(process_mask, region, task_id) for task_id, region in enumerate(masks_isolated)]
    books = [task.result() for task in tasks]

links = []

isbns = []
authors = []

def process_book_with_google_books(book):
    response = requests.get(
        f"https://www.googleapis.com/books/v1/volumes?q={book}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response = response.json()

    isbn, author, link = "NULL", "NULL", "NULL"

    try:
        isbn = response["items"][0]["volumeInfo"]["industryIdentifiers"][0]["identifier"]
        if (
            "volumeInfo" in response["items"][0]
            and "authors" in response["items"][0]["volumeInfo"]
        ):
            author = response["items"][0]["volumeInfo"]["authors"][0]
        link = response["items"][0]["volumeInfo"]["infoLink"]
    except:
        pass

    return isbn, author, link

with concurrent.futures.ThreadPoolExecutor() as executor:
    tasks = [executor.submit(process_book_with_google_books, book) for book in books]

    for task in tasks:
        isbn, author, link = task.result()
        isbns.append(isbn)
        authors.append(author)
        links.append(link)

with open("annotations.json", "w") as f:
    json.dump(
        [
            {
                "title": title,
                "author": author,
                "isbn": isbn,
                "polygons": [polygon.tolist() for polygon in polygon_list],
                "xyxy": xyxy.tolist(),
                "link": link,
            }
            for title, author, isbn, polygon_list, xyxy, link in zip(
                books, authors, isbns, polygons, results.xyxy, links
            ) if "sorry" not in title.lower() and "NULL" not in title
        ],
        f,
    )

with open("annotations.json", "r") as f:
    annotations = json.load(f)

width, height = image.shape[1], image.shape[0]


with open(args.output, "w") as f:
    f.write(
        f"""<div class="image-container"><img src="{args.image}" height="{height}" width="{width}">
        
<svg width="{width}" height="{height}">"""
    )
    for annotation in annotations:
        polygons = annotation["polygons"][0]
        f.write(
            f"""<polygon points="{', '.join([f'{x},{y}' for x, y in polygons])}" fill="transparent" stroke="red" stroke-width="2"
        onclick="window.location.href='{annotation['link']}';"></polygon>"""
        )
    f.write("</svg>")
    f.write(
        """
<style>
  .image-container {
    position: relative;
    height: HEIGHTpx;
    width: WIDTHpx;
  }
  img, svg {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: auto;
  }
  svg {
    z-index: 1;
  }
</style></div>""".replace(
            "HEIGHT", str(height)
        ).replace(
            "WIDTH", str(width)
        )
    )
