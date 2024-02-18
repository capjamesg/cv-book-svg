import base64
import json
import optparse

import cv2
import numpy as np
import requests
import supervision as sv
import tqdm
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from openai import OpenAI

parser = optparse.OptionParser()

parser.add_option("-i", "--image", dest="image", help="Image to process")
parser.add_option("-o", "--output", dest="output", help="Output file")

args, args2 = parser.parse_args()

image = cv2.imread(args.image)

base_model = GroundedSAM(ontology=CaptionOntology({"book spine": "book spine"}))

results = base_model.predict(image)

client = OpenAI()

masks_isolated = []

masks_to_xyxys = sv.mask_to_xyxy(masks=results.mask)
image = cv2.imread(image)

polygons = [sv.mask_to_polygons(mask) for mask in results.mask]

for mask in results.mask:
    masked_region = np.zeros_like(image)
    masked_region[mask] = image[mask]
    masks_isolated.append(masked_region)

books = []
links = []

for region in tqdm.tqdm(masks_isolated):
    region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite("region.jpeg", region)

    with open("region.jpeg", "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

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

    books.append(
        response.choices[0].message.content.rstrip("Title:").replace("\n", " ")
    )

isbns = []
authors = []

for book in books:
    response = requests.get(
        f"https://www.googleapis.com/books/v1/volumes?q={book}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response = response.json()

    try:
        isbns.append(
            response["items"][0]["volumeInfo"]["industryIdentifiers"][0]["identifier"]
        )
        if (
            "volumeInfo" in response["items"][0]
            and "authors" in response["items"][0]["volumeInfo"]
        ):
            authors.append(response["items"][0]["volumeInfo"]["authors"][0])
        else:
            authors.append("NULL")
        links.append(response["items"][0]["volumeInfo"]["infoLink"])
    except:
        isbns.append("NULL")
        authors.append("NULL")
        links.append("NULL")

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
            )
        ],
        f,
    )

with open("annotations.json", "r") as f:
    annotations = json.load(f)

annotated_image = cv2.imread(args.image)

width, height = annotated_image.shape[1], annotated_image.shape[0]


with open(args.output, "w") as f:
    f.write(
        f"""<div class="image-container"><img src="{args.image}" height="{height}" width="{width}">
        
<svg width="{width}" height="{height}">"""
    )
    for annotation in annotations:
        polygons = annotation["polygons"][0]
        print(polygons)
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
