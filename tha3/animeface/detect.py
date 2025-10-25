import cv2 # type: ignore
import os

def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
    """Detect faces in an image and return a list of rectangles.

    Returns:
        List of [x, y, w, h] for each detected face. When no faces are
        detected, returns an empty list.

    Args:
        filename: path to input image
        cascade_file: path to cascade xml
        show: if True, display the image with rectangles (same as before)
        out_path: if not None, save the output image with rectangles to this path
    """
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))

    # Convert to list of lists so it's easy to return/serialize
    coords = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces]

    return coords

