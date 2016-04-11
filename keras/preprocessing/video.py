""" Fairly basic set of tools for realtime data augmentation on video data.
It uses all the transformations related to image applied to each frame and
includes new utilities for video preprocessing.
Also provides a new generator for video data which can load video data from
files in parallel while the training is ongoing.
"""
from __future__ import absolute_import

import os
import re

import numpy as np

import cv2
from six.moves import range


def trim(x, start_frame=0, end_frame=None, length=None):
    """ Trim a video array on its temporal dimension
    Args:
        x (4D array): video array. Can be in 'th' or 'tf' dimension
            ordering
        start_frame (Optional[int]): Number of the frame to start to read
            the video
        end_frame (Optional[int]): Number of the frame to end reading the
            video.
        length (Optional[int]): Number of frames of length you want to read
            the video from the start_frame. This override the end_frame
            given before.
    """
    if x.shape[0] == 3:
        dim_ordering = 'th'
        x = x.transpose(1, 2, 3, 0)
    elif x.shape[3] == 3:
        dim_ordering = 'tf'
    else:
        raise Exception('Invalid input array')

    num_frames = x.shape[0]
    if start_frame >= num_frames or start_frame < 0:
        raise Exception('Invalid initial frame given')
    # Set up until which frame to read
    if end_frame:
        end_frame = end_frame if end_frame < num_frames else num_frames
    elif length:
        end_frame = start_frame + length
        end_frame = end_frame if end_frame < num_frames else num_frames
    else:
        end_frame = num_frames
    if end_frame < start_frame:
        raise Exception('Invalid ending position')

    x = x[start_frame:end_frame, :, :, :]
    if dim_ordering == 'th':
        x = x.transpose(3, 0, 1, 2)
    return x


def random_trim(x, length):
    """ Returns a trim of the video at a random temporal position of the video.
    The random trim is uniformly distributed along the video length.
    Args:
        x (4D array): video array.
        length (int): Number of frames of length to trim the video.
    """
    # Supose 'th' dimesion ordering
    total_length = x.shape[1]
    if length >= total_length:
        return x
    start = no.random.randint(total_length-length)
    return trim(x, start_frame=start, length=length)


def temporal_flip(x):
    """ Flip the temporal sequence of the video
    """
    return x[..., ::-1, :, :]


def video_to_array(video_path, resize=None, start_frame=0, end_frame=None,
                   length=None, dim_ordering='th'):
    """ Convert the video at the path given in to an array

    Args:
        video_path (string): path where the video is stored
        resize (Optional[tupple(int)]): desired size for the output video.
            Dimensions are: height, width
        start_frame (Optional[int]): Number of the frame to start to read
            the video
        end_frame (Optional[int]): Number of the frame to end reading the
            video.
        length (Optional[int]): Number of frames of length you want to read
            the video from the start_frame. This override the end_frame
            given before.
    Returns:
        video (nparray): Array with all the data corresponding to the video
                         given. Order of dimensions are: channels, length
                         (temporal), height, width.
    Raises:
        Exception: If the video could not be opened
    """
    if dim_ordering not in ('th', 'tf'):
        raise Exception('Invalid dim_ordering')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception('Could not open the video')

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame >= num_frames or start_frame < 0:
        raise Exception('Invalid initial frame given')
    # Set up the initial frame to start reading
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # Set up until which frame to read
    if end_frame:
        end_frame = end_frame if end_frame < num_frames else num_frames
    elif length:
        end_frame = start_frame + length
        end_frame = end_frame if end_frame < num_frames else num_frames
    else:
        end_frame = num_frames
    if end_frame < start_frame:
        raise Exception('Invalid ending position')

    frames = []
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            # The resize of CV2 requires pass firts width and then height
            frame = cv2.resize(frame, resize)
        frames.append(frame)

    video = np.array(frames, dtype=np.float32)
    if dim_ordering == 'th':
        video = video.transpose(3, 0, 1, 2)
    return video

def list_videos(directory, ext='mp4|avi'):
    """ List all videos stored on a directory
    """
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and \
            re.match(r'([\w]+\.(?:' + ext + '))', f)]
