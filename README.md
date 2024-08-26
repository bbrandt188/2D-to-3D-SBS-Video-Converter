# 2D to 3D SBS Video Converter

This is a bash script to take any 2D Video file you have and then process it into a 3D Side-By-Side video file, compatable with any player that supports 3D SBS Video Playback.

It's simplistic and will take the viddo, offset the frame rate slightly and then combine them together to give a 3d effect from the offset images. 

Works best when resulting file is viewed on VR Headsets otherwise It's very difficult to get the desired effect.
Si

## Requirements
- ffmpeg
- bash

## How To Use
Allow the file to be executable
`chmod +x sbs3d.sh`

Then run the file with input video file, 3d factor (Integer; Default is 20), and delay between frames (float; Can use decimal numbers; default is 1/FPS simply inputed as just the number ex: 1)
```bash
./sbs3d.sh $0 <video file> 20 1
``` 

This will output the completed video after processing with the same name as the input suffixed with `3dsbs-{3dFactor}-{frame_delay}.mp4` in the same folder as you ran it in 