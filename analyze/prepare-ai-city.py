import os
from moviepy.editor import VideoFileClip

def trim_videos_to_shortest(root_dir, video_ids_str, output_dir):
    """
    Trims a list of videos to the length of the shortest video among them.

    Args:
        root_dir (str): The root directory where your video dataset is located.
                        Example: 'root-dir'
        video_ids_str (str): A space-separated string of video identifiers.
                             Example: 'S05-c010 S05-c016 S01-c001'
        output_dir (str): The directory where the trimmed videos will be saved.
                          This directory will be created if it doesn't exist.
    """

    video_ids = video_ids_str.split()
    video_paths = {}
    frame_counts = {}

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output videos will be saved in: {output_dir}")

    print("\n--- Step 1: Checking video lengths ---")
    # First, get the frame count for each video
    for video_id in video_ids:
        try:
            # Parse the video ID to construct the path
            parts = video_id.split('-')
            if len(parts) != 2:
                print(f"Skipping invalid video ID format: {video_id}. Expected 'Section-camera'.")
                continue

            section_number = parts[0] # e.g., S05
            camera_number = parts[1]  # e.g., c010

            # Construct the full path
            # Assuming all input videos are named vdo.avi
            video_file_name = "vdo.avi"
            current_video_path = os.path.join(root_dir, section_number, camera_number, video_file_name)

            if not os.path.exists(current_video_path):
                print(f"Warning: Video not found at {current_video_path}. Skipping.")
                continue

            # Load the video clip to get its duration and FPS
            clip = VideoFileClip(current_video_path)
            duration_seconds = clip.duration
            fps = clip.fps
            frame_count = int(duration_seconds * fps) # Calculate total frames

            video_paths[video_id] = current_video_path
            frame_counts[video_id] = frame_count
            print(f"Video '{video_id}' ({current_video_path}): {frame_count} frames ({duration_seconds:.2f} seconds)")
            clip.close() # Close the clip to release resources

        except Exception as e:
            print(f"Error processing video {video_id} at {current_video_path}: {e}")
            if video_id in video_paths:
                del video_paths[video_id] # Remove from consideration if error
            if video_id in frame_counts:
                del frame_counts[video_id] # Remove from consideration if error

    if not frame_counts:
        print("No valid videos found or processed. Exiting.")
        return

    # Find the shortest frame count
    shortest_frame_count = min(frame_counts.values())
    print(f"\n--- Shortest video length found: {shortest_frame_count} frames ---")

    print("\n--- Step 2: Trimming videos ---")
    # Trim and save each video
    for video_id, original_path in video_paths.items():
        try:
            # Changed output extension to .mp4
            output_file_name = f"{video_id}.mp4"
            output_path = os.path.join(output_dir, output_file_name)

            print(f"Trimming '{video_id}' to {shortest_frame_count} frames and saving to '{output_path}'...")

            # Load the original clip
            clip = VideoFileClip(original_path)

            # Calculate the duration in seconds for the shortest frame count
            target_duration_seconds = min(shortest_frame_count / clip.fps, clip.duration)

            # Trim the clip from the beginning (0 seconds) to the target duration
            trimmed_clip = clip.subclip(0, target_duration_seconds)

            # Write the trimmed clip to the output file as MP4
            # `codec='libx264'` is a common and good choice for MP4.
            # `fps=clip.fps` maintains the original frame rate.
            trimmed_clip.write_videofile(output_path, codec="libx264", fps=clip.fps)

            print(f"Successfully trimmed and saved: {output_file_name}")
            clip.close() # Close original clip
            trimmed_clip.close() # Close trimmed clip

        except Exception as e:
            print(f"Error trimming video {video_id}: {e}")

    print("\n--- Video trimming process completed ---")


# --- How to use the script ---
if __name__ == "__main__":
    # IMPORTANT: Replace 'path/to/your/root-dir' with the actual path to your root directory
    ROOT_DIRECTORY = '/ssd0/datasets/AIC/AIC22/validation'

    # IMPORTANT: Replace this with your actual list of video identifiers
    VIDEO_IDENTIFIERS = 'S05-c010 S05-c016 S05-c018 S05-c017 S05-c020 S05-c022 S05-c023 S05-c026 S05-c027'

    # IMPORTANT: Define your desired output directory here
    # For example: '/Users/yourusername/Documents/trimmed_videos'
    OUTPUT_DIRECTORY = '/ssd0/datasets/pipeline-scheduler/AIC'

    # Call the function to start the process
    trim_videos_to_shortest(ROOT_DIRECTORY, VIDEO_IDENTIFIERS, OUTPUT_DIRECTORY)
