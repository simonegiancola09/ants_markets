from PIL import Image
from PIL import ImageFile
import glob
import re
 
# Create the frames
def GIF_creator(directory_source, filename, directory_destination, duration):
    frames = []
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # access a list of pngs in a folder
    imgs = sorted(glob.glob(directory_source + '*.png'), key=extract_number)
    for i in imgs:
        # create frame list
        new_frame = Image.open(i)
        frames.append(new_frame)
    # create GIF
    skip = 10
    frames[0].save(directory_destination + filename + '.gif',
                format='GIF',
                append_images=frames[1::skip],
                save_all=True,
                duration=duration, loop=0)

def extract_number(file_name):
    # Extract the numeric part using regular expression
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group())
    return 0  # Default value if no number found