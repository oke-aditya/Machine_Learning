import os
import cv2
import face_recognition
from tqdm import tqdm

KNOWN_FACES_DIR = r'known_images'
UNKNOWN_FACES_DIR = r"uknown_images"
# Degree of tolerance for false positives
# Lower tolereance for getting perfect matches.
# Default = 0.6

TOLERENCE = 0.6
# Frame drawing
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"   # default is hog

print("Loading Known faces")

known_faces = []
known_names = []

def name_to_color(name):
    color_list = []
    for c in name[:3]:
        color = (ord(c.lower()) - 97) * 8
        color_list.append(color)
    return color_list

# Finds the encoding of the known face which will elable us to recgnize
for name in tqdm(os.listdir(KNOWN_FACES_DIR)):
    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):
        image = face_recognition.load_image_file(os.path.join(os.path.join(KNOWN_FACES_DIR, name), filename))
        encoding = face_recognition.face_encodings(image)[0]          # Encode only the first face. Single shot
        # We don't need to know where the knwon faces are here !!!!!!
        known_faces.append(encoding)
        known_names.append(name)

print("Processing unkonwn faces")

for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = face_recognition.load_image_file(os.path.join(UNKNOWN_FACES_DIR, filename))
    locations = face_recognition.face_locations(image, model=MODEL)
    # Get the locations where the image is found
    encodings = face_recognition.face_encodings(image, locations)
    # Get the encoding where we located the face
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print("Found %d number of faces " % (len(encodings)))

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance=TOLERENCE)
        print(results)
        # Returns a list of Booleans denoting the index where it was matched
        # match = None
        if True in results:
            match = known_names[results.index(True)]  # Just getting the one single identity
            print("Match found: %s" % (match))

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2] + 22)  # Slightly up

            # color = [0, 255, 0]
            color = name_to_color(match)

            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            cv2.putText(image, match, (face_location[3] + 10, face_location[0] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(200, 200, 200), thickness=FONT_THICKNESS)

    cv2.imshow(filename, image)
    cv2.waitKey(10000) & 0xFF == ord('q')
    cv2.destroyAllWindows()
