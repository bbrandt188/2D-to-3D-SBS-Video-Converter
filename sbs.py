import os
import sys
import cv2
import mlx.core as mx
import numpy as np
import subprocess
import tempfile
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Set MLX default device to GPU to use Apple Metal
mx.set_default_device(mx.Device(mx.DeviceType.gpu))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(messages)s')

def process_batch_with_mlx(frames_batch):
    """Process a batch of video frames using MLX for depth mapping"""
    mlx_array = mx.array(np.array(frames_batch, dtype=np.float32))
    grayscale_array = mx.mean(mlx_array, axis=3)
    processed_array = mx.multiply(grayscale_array, 1.5)
    
    # Convert MLX array back to NumPy using buffer protocol
    processed_frames = np.array(processed_array).astype(np.uint8)
    return processed_frames

def apply_depth_to_frame(frame, depth_map, max_shift=20):
    height, width = frame.shape[:2]
    
    # Vectorized depth-based shift calculation
    shift = ((depth_map / 255.0) * max_shift).astype(int)
    
    x_indices = np.arrange(width)
    left_x_indices = np.clip(x_indices[None, :] - shift, 0, width - 1)
    right_x_indices = np.clip(x_indices[None, :] + shift, 0, width - 1)
    
    # Create left & right eye images
    left_eye = frame(np.arange(height)[:, None], left_x_indices)
    right_eye = frame(np.arange(height)[:, None], right_x_indices)      
    
    return left_eye, right_eye              

def process_and_save_frames(batch_index, frames_batch, temp_dir, max_shift):
    """Process and save a batch of frames"""
    depth_maps_batch = process_batch_width_mlx(frames_batch)
    frame_paths = []
    
    for i, (frame, depth_map) in enumerate(zip(frames_batch, depth_maps_batch)):
        left_eye, right_eye = apply_depth_to_frame(frame, depth_map, max_shfit)
        sbs_frame = np.hstack((left_eye, right_eye))
        frame_path = ox.path.join(temp_dir, f"sbs_frame_{batch_index}_{i:04d}.png")
        cv2.imwrite(frame_path, sbs_frame)
        frame_paths.append(f"file '{frame_path}'\n")
        
    return frame_paths

def process_video(input_video, output_dir, delay, max_shift=20, batch_size=10):
    """Process the video into side-by-side frames with depth mapping"""
    video_capture = cv2.VideoCapture(input_video)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fpx = video_capture.get(cv2.CAP_PROP_FPS)
    
    batches = (frame_count + batch_size - 1) // batch_size
    pool = Pool(cpu_count())
    
    with tempfile.TemporaryDirectory() as temp_dir:
        all_frame_paths = []
        
        for batch_index in tqdm(range(batches)):
            frames_batch = []
            for _ in range(batch_size):
                ret, frame = video_capture.read()
                if not ret:
                    break
                frames_batch.append(frame)
            
            if not frames_batch:
                break
            
            frame_paths = pool.apply_async(process_and_save_frames, (batch_index, frames_batch, temp_dir, max_shift)).get()
            all_frame_paths.extend(frame_paths)
            
        pool.close()
        pool.join()
        
        # Save the final video
        output_video_path = os.path.join(output_dir, 'output_video.mp4')
        with open(os.path.join(temp_dir, 'frames_list.txt'), 'w') as f:
            f.writelines(all_frame_paths)
            
        intermediate_video = os.path.join(temp_dir, 'intermediate.mp4')
        
        
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', os.path.join(temp_dir, 'frames_list.txt'), # input frames list
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', # Encode to H.264
            intermediate_video # output intermediate video
        ])

        subprocess.run([
            'ffmpeg', '-y', '-i', intermediate_video, # intermediate video created from frames
            '-i', input_video,
            '-filter_complex', 
        ])

subprocess.run([
            'ffmpeg', '-y', '-i', os.path.join(temp_dir, 'frames_list.txt'),  # input frames list
            '-i', input_video,  # input original video for audio
            '-filter_complex', '[0:v]split=2[left][right]; \
                                [left]crop=in_w/2:in_h:0:0[left_eye]; \
                                [right]crop=in_w/2:in_h:in_w/2:0[right_eye]; \
                                [left_eye][right_eye]hstack=inputs=2[sbs]',
            '-map', '[sbs]',  # map the combined video
            '-map', '1:a?',  # map the original audio
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',  # encode video as H.264
            '-c:a', 'copy',  # copy the audio
            os.path.join(output_dir, 'output_video.mp4')
        ])


        #subprocess.run([
            #'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-fflags', '+genpts', '-use_wallclock_as_timestamps', '1', '-i',
            #os.path.join(temp_dir, 'frames_list.txt'), '-pix_fmt', 'yuv420p', '-c:v', 'libx264', 
            #output_video_path
        #])

    video_capture.release()

if __name__ == "__main__":
    input_video = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    max_shift = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    process_video(input_video, output_dir, delay, max_shift)
