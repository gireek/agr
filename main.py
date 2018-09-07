from age_detection import age_detection
import argparse
from moviepy.editor import VideoFileClip

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--live", type=int, default=0,
                         help="to get a live mode enter 1")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    depth = args.depth
    width = args.width
    live = args.live
    age_detect = age_detection.AgeDetection(depth=depth, width=width)
    if not live:
        # face_detect = FaceDetection(save_image_flag=False)
        vid_output = "output5.mp4"
        subclip_start = '00:00:05.00'
        subclip_end = '00:0:05.50'
        clip = VideoFileClip("./input_video/VID_20180822_135016.mp4").subclip(subclip_start, subclip_end)
        vid = clip.fl_image(age_detect.pipeline)
        vid.write_videofile(vid_output, audio=False)
        age_detect.create_metadata()
    else:
        age_detect.live_from_video('./input_video/VID_20180822_135016.mp4')
    #print(age_detect.people_dict)


if __name__ == "__main__":
    main()
