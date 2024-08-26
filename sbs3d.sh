#!/bin/bash

# Thanks to MDocumento & threedguru's script found here:
#     https://github.com/threedguru/threedguru/blob/main/cv3DSBS.cmd

if [ -z "$1" ]; then
    echo "Usage: $0 <video file> OPTIONS"
    echo "    OPTIONS:"
    echo "        3dfactor - the 3D factor between left and right frames."
    echo "                   Default value: 20"
    echo "        delay    - delay between left and right frames and adds"
    echo "                   illusion of depth"
    echo "                   Default value: 1/FPS of input video"
    echo
    echo "Dependencies: ffmpeg ffprobe"
    exit 1
fi

# Get input file
inp="$1"

# Get 3d factor
if [ -z "$2" ]; then
    factor_3d=20
else
    factor_3d=$2
fi

# Delaying right frames adds illusion of depth
if [ -z "$3" ]; then
    # Get frame rate of input video
    fps=$(ffprobe -loglevel quiet -hide_banner -show_streams \
        -select_streams v "$inp" | \
        grep "^r_frame_rate" | \
        sed "s/^r_frame_rate=\(.*\)$/\1/g")
            right_delay=`bc -l <<< "1/($fps)"`
            right_delay="0$right_delay"
else
    right_delay=$3
fi

# Create output filename
ibasename=`basename "$inp"`
outp="$ibasename,3dsbs-$factor_3d-${right_delay:0:6}.mp4"

# Get width and height of input video
width=$(ffprobe -loglevel quiet -hide_banner -show_streams -select_streams v \
    "$inp" | grep "^width" | sed "s/^width=\([0-9]*\).*$/\1/g")
height=`ffprobe -loglevel quiet -hide_banner -show_streams -select_streams v \
    "$inp" | grep "^height" | sed "s/^height=\([0-9]*\).*$/\1/g"`

new_width=$((width + width / factor_3d))
new_height=$((height + height / factor_3d))
crop_width=$(((new_width - width) / 2))

echo "New Resolution: $new_width X $new_height"
echo "3D Factor     : $factor_3d"
echo "Delay         : $right_delay (FPS: $fps)"

# Use FFmpeg to create the 3D SBS Video
ffmpeg -i "$inp" -filter_complex \
"[0:v]scale=$new_width:$new_height,crop=$width:$new_height:$crop_width:0[left];
[0:v]scale=$new_width:$new_height,crop=$width:$new_height:0:0,
setpts=(PTS-$right_delay/TB)[right];
[left][right]hstack" \
-c:v libx264 -preset veryfast -crf 28 "$outp"

# Print Success Message
echo "3D SBS video created at: $outp"