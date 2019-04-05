import args
import datetime
import logging
import os
from os import path
import subprocess
import shutil
import shlex
import tempfile
import time

import cv2
import numpy as np
from openvino import inference_engine as ie

import openvino_detection
import utils


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def get_parser():
    parser = args.base_parser('Test movidius')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for detecting faces',
    )
    parser.add_argument(
        '--video',
        help='Path to the source video file to be processed (or URL to camera).',
    )
    parser.add_argument(
        '--output',
        help='Path to the output (processed) video file to write to.',
    )
    parser.add_argument(
        '--sound',
        help='include sound in the output video.',
        default=False,
        action='store_true'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    facenet = openvino_detection.OpenVINOFacenet(args)

    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Remove file if any
    try:
        os.remove(args.output)
    except:
        pass

    # Read codec information from input video.
    ex = int(video.get(cv2.CAP_PROP_FOURCC))
    codec = (
            chr(ex & 0xFF) +
            chr((ex & 0xFF00) >> 8) +
            chr((ex & 0xFF0000) >> 16) +
            chr((ex & 0xFF000000) >> 24)
    )

    # cuda = built_with_cuda()
    # if not cuda:
    #     codec = 'MP4V'

    print('Create video %sx%s with FPS %s and CODEC=%s' % (width, height, fps, codec))
    fourcc = cv2.VideoWriter_fourcc(*codec)
    output_movie = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    frame_number = 0

    def get_frame():
        _, new_frame = video.read()

        if new_frame is None:
            print("frame is None.")
            return None

        # if new_frame.shape[0] > 480:
        #     new_frame = utils.image_resize(new_frame, height=480)

        return new_frame

    try:
        while True:
            # Capture frame-by-frame
            frame = get_frame()
            if frame is None:
                break
            # BGR -> RGB
            # rgb_frame = frame[:, :, ::-1]
            # rgb_frame = frame

            facenet.process_frame(frame)

            if frame_number % 10 == 0:
                # Write the resulting image to the output video file
                print("Writing frame {} / {}".format(frame_number, length))

            output_movie.write(frame)

            frame_number += 1

    except (KeyboardInterrupt, SystemExit) as e:
        print('Caught %s: %s' % (e.__class__.__name__, e))

    # When everything is done, release the capture
    video.release()
    cv2.destroyAllWindows()
    output_movie.release()
    print('Finished')

    logging.info('End video processing: %s', datetime.datetime.now())

    if args.sound:
        time.sleep(0.2)
        logging.info('Start merge audio: %s', datetime.datetime.now())
        merge_audio_with(args.video, args.output)
        logging.info('End merge audio: %s', datetime.datetime.now())

    import sys
    sys.exit(0)


def merge_audio_with(original_video_file, target_video_file):
    dirname = tempfile.gettempdir()
    audio_file = path.join(dirname, 'audio')

    # Get audio codec
    # cmd = (
    #     'ffprobe -show_streams -pretty %s 2>/dev/null | '
    #     'grep codec_type=audio -B 5 | grep codec_name | cut -d "=" -f 2'
    #     % original_video_file
    # )
    # codec_name = subprocess.check_output(["bash", "-c", cmd]).decode()
    # codec_name = codec_name.strip('\n ')
    # audio_file += ".%s" % codec_name
    audio_file += ".%s" % "mp3"

    # Something wrong with original audio codec; use mp3
    # -vn -acodec copy file.<codec-name>
    cmd = 'ffmpeg -y -i %s -vn %s' % (original_video_file, audio_file)
    code = subprocess.call(shlex.split(cmd))
    if code != 0:
        raise RuntimeError("Failed run %s: exit code %s" % (cmd, code))

    # Get video offset
    cmd = (
        'ffprobe -show_streams -pretty %s 2>/dev/null | '
        'grep codec_type=video -A 28 | grep start_time | cut -d "=" -f 2'
        % original_video_file
    )
    video_offset = subprocess.check_output(["bash", "-c", cmd]).decode()
    video_offset = video_offset.strip('\n ')

    # Get audio offset
    cmd = (
        'ffprobe -show_streams -pretty %s 2>/dev/null | '
        'grep codec_type=audio -A 28 | grep start_time | cut -d "=" -f 2'
        % original_video_file
    )
    audio_offset = subprocess.check_output(["bash", "-c", cmd]).decode()
    audio_offset = audio_offset.strip('\n ')

    dirname = tempfile.gettempdir()
    video_file = path.join(dirname, 'video')

    # Get video codec
    cmd = (
        'ffprobe -show_streams -pretty %s 2>/dev/null | '
        'grep codec_type=video -B 5 | grep codec_name | cut -d "=" -f 2'
        % original_video_file
    )
    codec_name = subprocess.check_output(["bash", "-c", cmd]).decode()
    codec_name = codec_name.strip('\n ')
    video_file += ".%s" % codec_name

    shutil.copyfile(target_video_file, video_file)
    # subprocess.call(["cp", target_video_file, video_file])
    time.sleep(0.2)

    cmd = (
        'ffmpeg -y -itsoffset %s -i %s '
        '-itsoffset %s -i %s -c copy %s' %
        (video_offset, video_file, audio_offset, audio_file, target_video_file)
    )

    code = subprocess.call(shlex.split(cmd))
    if code != 0:
        raise RuntimeError("Saving video with sound failed: exit code %s" % code)





if __name__ == "__main__":
    main()
