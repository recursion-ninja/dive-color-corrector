#!/opt/homebrew/bin/python

import argparse 
import cv2
import fnmatch
import math
import numpy as np
import io
import os
import re
import sys
import ctypes
import tempfile
import tqdm
import time
from contextlib import redirect_stderr, redirect_stdout, contextmanager
from itertools import groupby

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

VERSION_STR = '1.0.0'

TQDM_FORMAT = '{desc:35} {percentage: 5.1f}%|{bar:20}{r_bar}'
l_bar='{desc} {ncols:30}%|'
r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]' 

THRESHOLD_RATIO = 2000
MIN_AVG_RED = 60
MAX_HUE_SHIFT = 120
BLUE_MAGIC_VALUE = 1.2
SAMPLE_SECONDS = 2 # Extracts color correction from every N seconds

def hue_shift_red(mat, h):

    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)

    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]

    return np.dstack([r, g, b])

def normalizing_interval(array):

    high = 255
    low = 0
    max_dist = 0

    for i in range(1, len(array)):
        dist = array[i] - array[i-1]
        if(dist > max_dist):
            max_dist = dist
            high = array[i]
            low = array[i-1]

    return (low, high)

def apply_filter(mat, filt):

    r = mat[..., 0]
    g = mat[..., 1]
    b = mat[..., 2]

    r = r * filt[0] + g*filt[1] + b*filt[2] + filt[4]*255
    g = g * filt[6] + filt[9] * 255
    b = b * filt[12] + filt[14] * 255

    filtered_mat = np.dstack([r, g, b])
    filtered_mat = np.clip(filtered_mat, 0, 255).astype(np.uint8)

    return filtered_mat

def get_filter_matrix(mat):

    mat = cv2.resize(mat, (256, 256))

    # Get average values of RGB
    avg_mat = np.array(cv2.mean(mat)[:3], dtype=np.uint8)

    # Find hue shift so that average red reaches MIN_AVG_RED
    new_avg_r = avg_mat[0]
    hue_shift = 0
    while(new_avg_r < MIN_AVG_RED):

        shifted = hue_shift_red(avg_mat, hue_shift)
        new_avg_r = np.sum(shifted)
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT:
            new_avg_r = MIN_AVG_RED

    # Apply hue shift to whole image and replace red channel
    shifted_mat = hue_shift_red(mat, hue_shift)
    new_r_channel = np.sum(shifted_mat, axis=2)
    new_r_channel = np.clip(new_r_channel, 0, 255)
    mat[..., 0] = new_r_channel

    # Get histogram of all channels
    hist_r = hist = cv2.calcHist([mat], [0], None, [256], [0,256])
    hist_g = hist = cv2.calcHist([mat], [1], None, [256], [0,256])
    hist_b = hist = cv2.calcHist([mat], [2], None, [256], [0,256])

    normalize_mat = np.zeros((256, 3))
    threshold_level = (mat.shape[0]*mat.shape[1])/THRESHOLD_RATIO
    for x in range(256):

        if hist_r[x] < threshold_level:
            normalize_mat[x][0] = x

        if hist_g[x] < threshold_level:
            normalize_mat[x][1] = x

        if hist_b[x] < threshold_level:
            normalize_mat[x][2] = x

    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])


    shifted = hue_shift_red(np.array([1, 1, 1]), hue_shift)
    shifted_r, shifted_g, shifted_b = shifted[0][0]

    red_gain = 256 / (adjust_r_high - adjust_r_low)
    green_gain = 256 / (adjust_g_high - adjust_g_low)
    blue_gain = 256 / (adjust_b_high - adjust_b_low)

    redOffset = (-adjust_r_low / 256) * red_gain
    greenOffset = (-adjust_g_low / 256) * green_gain
    blueOffset = (-adjust_b_low / 256) * blue_gain

    adjust_red = shifted_r * red_gain
    adjust_red_green = shifted_g * red_gain
    adjust_red_blue = shifted_b * red_gain * BLUE_MAGIC_VALUE

    return np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        0, green_gain, 0, 0, greenOffset,
        0, 0, blue_gain, 0, blueOffset,
        0, 0, 0, 1, 0,
    ])

def correct(mat):
    original_mat = mat.copy()

    filter_matrix = get_filter_matrix(mat)

    corrected_mat = apply_filter(original_mat, filter_matrix)
    corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)

    return corrected_mat

def correct_image(input_path, output_path):
    mat = cv2.imread(input_path)
    rgb_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

    corrected_mat = correct(rgb_mat)

    cv2.imwrite(output_path, corrected_mat)

    preview = mat.copy()
    width = preview.shape[1] // 2
    preview[::, width:] = corrected_mat[::, width:]

    preview = cv2.resize(preview, (960, 540))

    return cv2.imencode('.png', preview)[1].tobytes()


def analyze_video(input_video_path, output_video_path):

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "ignore_chapters;1"

    # Initialize new video writer
    cap = cv2.VideoCapture(input_video_path, apiPreference=cv2.CAP_FFMPEG)
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    frame_count = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get filter matrices for every 10th frame
    filter_matrix_indexes = []
    filter_matrices = []
    count = 0

    with tqdm.tqdm(desc='          ⮑  Analyzing', position=1, unit=' frame', leave=False, bar_format=TQDM_FORMAT) as pbar:
        while(cap.isOpened()):
            pbar.update(1)
            count += 1  
            ret, frame = cap.read()
            if not ret:
                # End video read if we have gone beyond reported frame count
                if count >= frame_count:
                    break
     
                # Failsafe to prevent an infinite loop
                if count >= 1e6:
                    break
    
                # Otherwise this is just a faulty frame read, try reading next frame
                continue
            # Pick filter matrix from every N seconds
            if count % (fps * SAMPLE_SECONDS) == 0:
                mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                filter_matrix_indexes.append(count)
                filter_matrices.append(get_filter_matrix(mat))
        
            yield count         

    cap.release()

    # Build a interpolation function to get filter matrix at any given frame
    filter_matrices = np.array(filter_matrices)

    yield {
        "input_video_path": input_video_path,
        "output_video_path": output_video_path,
        "fps": fps,
        "frame_count": count,
        "filters": filter_matrices,
        "filter_indices": filter_matrix_indexes
    }


def process_video(video_data, yield_preview=False):

    cap = cv2.VideoCapture(video_data["input_video_path"])

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_video = cv2.VideoWriter(video_data["output_video_path"], fourcc, video_data["fps"], (int(frame_width), int(frame_height)))

    filter_matrices = video_data["filters"]
    filter_indices = video_data["filter_indices"]

    filter_matrix_size = len(filter_matrices[0])
    def get_interpolated_filter_matrix(frame_number):

        return [np.interp(frame_number, filter_indices, filter_matrices[..., x]) for x in range(filter_matrix_size)]

    frame_count = video_data["frame_count"]

    count = 0
    cap = cv2.VideoCapture(video_data["input_video_path"])
 
    with tqdm.tqdm(desc='          ⮑  Processing', position=1, total=frame_count, unit=' frame', leave=False, bar_format=TQDM_FORMAT) as pbar:
        while(cap.isOpened()):
            pbar.update(1)
            count += 1
            ret, frame = cap.read()
    
            if not ret:
                # End video read if we have gone beyond reported frame count
                if count >= frame_count:
                    break
    
                # Failsafe to prevent an infinite loop
                if count >= 1e6:
                    break
    
                # Otherwise this is just a faulty frame read, try reading next
                continue
    
            # Apply the filter
            rgb_mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            interpolated_filter_matrix = get_interpolated_filter_matrix(count)
            corrected_mat = apply_filter(rgb_mat, interpolated_filter_matrix)
            corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)
    
            new_video.write(corrected_mat)
    
            if yield_preview:
                preview = frame.copy()
                width = preview.shape[1] // 2
                height = preview.shape[0] // 2
                preview[::, width:] = corrected_mat[::, width:]
    
                preview = cv2.resize(preview, (width, height))
    
                yield percent, cv2.imencode('.png', preview)[1].tobytes()
            else:
                yield None



    cap.release()
    new_video.release()


def collect_files(path, pattern):
    reg_expr = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    matching = []
    for root, dirs, files in os.walk(path, topdown=True):
        matching += [os.path.join(root, i) for i in files if re.match(reg_expr, i)]

    return matching


def collate_files(files, output_dir_suffix, output_format):

    files_paired = []
    for fm in files:
        files_paired.append((fm, os.path.dirname(fm), os.path.basename(fm)))

    files_sorted = sorted(files_paired, key = lambda x : (x[1],x[2]), reverse=False)
    files_marked = []
    for num, fs in enumerate(files_sorted):
        out_dir = fs[1] + output_dir_suffix
        out_ext = os.path.splitext(fs[2])[1].lower()
        out_file = output_format.format(num + 1)
        out_path = os.path.join(out_dir, out_file + out_ext)
        files_marked.append((fs[0], out_path))

    return files_marked


def interpret_path(path):
    return os.path.abspath(os.path.expanduser(path))


def interpret_target_directory(path):
    fpath = interpret_path(path)
    dpath = fpath if os.path.isdir(fpath) else os.path.dirname(fpath)
    bname = os.path.basename(dpath)
    dname = bname if bname else os.path.dirname(in_path)
    return (dpath, dname)


def gather_images(full_path, main_dir):

    found = collect_files(full_path, "*.jpg")
    pairs = collate_files(found, '-CORRECTED', main_dir + '-{:04d}')
    return pairs


def gather_videos(full_path, main_dir):

    found = collect_files(full_path, "*.mov")
    pairs = collate_files(found, '-CORRECTED', main_dir + '-{:04d}')
    return pairs

@contextmanager
def suppress_FFmpeg():
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            with redirect_stderr(null):
                yield


@contextmanager
def stderr_redirector(stream):
    original_stderr_fd = sys.stderr.fileno()

    def _redirect_stderr(to_fd):
        libc.fflush(c_stderr)
        sys.stderr.close()
        os.dup2(to_fd, original_stderr_fd)
        sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, 'wb'))

    saved_stderr_fd = os.dup(original_stderr_fd)
    try:
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stderr(tfile.fileno())
        yield
        _redirect_stderr(saved_stderr_fd)
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read().decode())
    finally:
        tfile.close()
        os.close(saved_stderr_fd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    description='📷 Underwater Media Color Correction\n⭕ Adds back the missing red light to underwater images and videos!',
                    epilog='🌊🏝️ \n🤿🐠\n🪸🦀',
                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filepath', metavar='DIR', help='Path containing media') #, required=True)
    parser.add_argument('--version', action='version', version='%(prog)s ' + VERSION_STR)

    if len(sys.argv) <= 1:
        parser.print_help()
        exit()

    user_input = parser.parse_args()
    path_given = interpret_path(user_input.filepath)

    if not os.path.exists(path_given):
        exit("The supplied file path does not exist:\n\t" + path_given)

    (full_path, main_dir) = interpret_target_directory(path_given)
    images = gather_images(full_path, main_dir)
    videos = gather_videos(full_path, main_dir)

#    for x in images:
#        print(x[0], x[1])

    total_images = len(images)
    total_videos = len(videos)
    print('🌊 Underwater Media Color Correction')
    print("🔍 Found:")
    print("📷   Images:", total_images)
    print("🎞    Videos:", total_videos)
    print()

    # Ensure the user wishes to proceed
    char_given = None
    while char_given not in ['y', 'n']:
        char_given = input('✋ Do you want to continue? [Y]es/[N]o: ')
        char_given = None if not char_given else char_given[0].lower()
    if char_given == 'n':
        exit("\n❌ Color correction aborted.")

    # TODO: check for folder!
        
    print('👌 Correcting color of:')
    with tqdm.tqdm(images, unit='image', bar_format=TQDM_FORMAT) as pbar_images:
        for (file_original, file_altered) in pbar_images:
            file_name = os.path.basename(file_original)
            pbar_images.set_description_str("    Image 𐎚 '%s'" % file_name)
            mat = cv2.imread(file_original)
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
            corrected_mat = correct(mat)
            cv2.imwrite(file_altered, corrected_mat)

    with tqdm.tqdm(videos, unit='video', bar_format=TQDM_FORMAT, position=0) as pbar_videos:
        for (file_original, file_altered) in pbar_videos:
            file_name = os.path.basename(file_original)
            pbar_videos.set_description_str("    Video 𐎚 '%s' " % file_name)
            for item in analyze_video(file_original, file_altered):
                if type(item) == dict:
                    video_data = item
            [x for x in process_video(video_data, yield_preview=False)]
