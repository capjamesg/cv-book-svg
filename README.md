# Make your bookshelf clickable

Use computer vision to generate an SVG that you can overlay onto a photo of your bookshelf that lets you click on each book to find out more information.

## Demo

[Try the demo](https://capjamesg.github.io/cv-book-svg/annotations.html)

https://github.com/capjamesg/cv-book-svg/assets/37276661/ec57bf18-4182-4dce-870f-6bef81809e80

## How it Works

This tool uses computer vision to identify and segment each book spine in an image of a bookshelf. Then, each book spine is sent to GPT-4 with Vision to read the book title and, if possible, the author.

This information is then sent to the Google Books API. The book ISBN, author name, and other meta information is retrieved from this API.

An SVG is then created using the segmented book spines. Each book is assigned a polygon which, when clicked, takes you to the Google Books page associated with a book.

This script uses the following vision tools:

- Grounding DINO (zero-shot object detection model)
- Segment Anything (image segmentation model)
- GPT-4 with Vision API
- OpenCV Python

It takes around 20 seconds to generate the polygons that map to the location of each book on an M1 Macbook Air. It then takes a few seconds to process each book with the OpenAI GPT-4 with Vision API.

For a bookshelf with 11 books, the script takes around one minute to run.

The script returns a HTML file with an SVG file that is overlaid onto the source image.

## How to Use

First, clone this project and install the required dependencies:

```
git clone https://github.com/capjamesg/cv-book-svg
cd cv-book-svg
pip3 install -r requirements.txt
```

Then, run the main script:

```
python3 grounded.py --image=example.jpg --output=annotation.html
```

This script takes an image as input (PNG, JPEG) and outputs a HTML document.

## Limitations

This system may:

- Not identify all books on a bookshelf (thin books are more likely to not be identified).
- Generate a link to the wrong Google Books URL (which will happen if a book is not available on Google Books, or if a book has a generic title like "Poems of Emily Dickinson", which could on its own refer to several publications).
- Mis-identify some books.

## License

This project is licensed under an [MIT license](LICENSE).

## Contributing

Found a bug? Have an idea that you'd like to see in the project? Open an Issue in this GitHub repository.
