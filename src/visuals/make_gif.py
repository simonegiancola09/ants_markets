from PIL import Image
import glob
 
# Create the frames
def GIF_creator(directory_source, filename, directory_destination):
    frames = []
    # access a list of pngs in a folder
    imgs = sorted(glob.glob(directory_source + '*.png'))
    for i in imgs:
        # create frame list
        new_frame = Image.open(i)
        frames.append(new_frame)
    # create GIF
    frames[0].save(directory_destination + filename + '.gif',
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=300, loop=0)