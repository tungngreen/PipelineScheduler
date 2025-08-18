import os
import numpy as np
import argparse
import rosbag #pip install bagpy
import cv2

def save_video(bag, output_file, target_topic, frame_rate=30.0):
    print(f"Starting to process images from topic: {target_topic}")
    video_writer = None
    image_size = None
    frame_count=0
    try:
        for topic, msg, t in bag.read_messages(topics=[target_topic]):
            if msg._type == 'sensor_msgs/Image':
                # Convert ROS Image message to OpenCV image
                cv_image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
                if msg.encoding == 'bgr8':
                    cv_image = cv_image[:, :, ::-1]
            elif msg._type == 'sensor_msgs/CompressedImage':
                # Convert ROS CompressedImage message to OpenCV image
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                continue

            if image_size is None:
                # Initialize video writer with the size of the first image
                height, width, _ = cv_image.shape
                image_size = (width, height)
                print(f"Detected image size: {image_size}")

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, image_size)
                if not video_writer.isOpened():
                    print(f"Error: Could not open video writer for file '{output_file}'.")
                    bag.close()
                    return

            if video_writer is not None:
                video_writer.write(cv_image)
                video_writer.write(cv_image)
                frame_count += 1

    finally:
        print(f"Finished processing. Total frames processed: {frame_count}")
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {output_file}")


def extract_bag_to_video(dataset, bag_file, output_file):
    """
    Extracts image frames from a ROS bag file and saves them to an MP4 video.

    Args:
        bag_file (str): The path to the input ROS bag file.
        output_file (str): The path for the output MP4 video file.
    """
    if not os.path.exists(bag_file):
        print(f"Error: The bag file '{bag_file}' does not exist.")
        return

    print(f"Opening bag file: {bag_file}")
    try:
        bag = rosbag.Bag(bag_file, 'r')
    except rosbag.ROSBagException as e:
        print(f"Error opening bag file: {e}")
        return

    image_topics = []
    print("Searching for image topics...")
    for topic, msg, t in bag.read_messages():
        if msg._type == 'sensor_msgs/CompressedImage':
            if topic not in image_topics:
                image_topics.append(topic)
                print(f"Found compressed image topic: {topic}")
        elif msg._type == 'sensor_msgs/Image':
            if topic not in image_topics:
                image_topics.append(topic)
                print(f"Found image topic: {topic}")

    if not image_topics:
        print("No 'sensor_msgs/Image' topics found in the bag file. Exiting.")
        bag.close()
        return

    if dataset == 'TorWIC':
        print("Dataset is TorWIC, using specific image topics.")
        if '/front/realsense/color/image_raw' in image_topics:
            save_video(bag, output_file, '/front/realsense/color/image_raw', 15)
        else:
            print("No suitable image topic found for TorWIC dataset. Available topics:")
            for topic in image_topics:
                print(f" - {topic}")
    elif dataset == 'SCAND':
        print("Dataset is SCAND, using specific image topics.")
        if '/camera/rgb/image_raw/compressed' in image_topics:
            save_video(bag, output_file, '/camera/rgb/image_raw/compressed', 7)
        elif '/spot/camera/frontright/image/compressed' in image_topics:
            save_video(bag, output_file, '/spot/camera/frontright/image/compressed', 7)
            save_video(bag, output_file.replace('.mp4', '_back.mp4'), '/spot/camera/back/image/compressed', 7)
        else:
            print("No suitable image topic found for SCAND. Available topics:")
            for topic in image_topics:
                print(f" - {topic}")

    bag.close()
    print("Bag file closed.")


def extract_images_to_video(input_dir, output_file):
    images = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if not images:
        print(f"No images found in directory: {input_dir}")
        return
    images.sort()
    first_image = cv2.imread(os.path.join(input_dir, images[0]))
    if first_image is None:
        print(f"Error reading the first image in directory: {input_dir}")
        return
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, 15.0, (width, height))
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for file '{output_file}'.")
        return
    for image in images:
        img_path = os.path.join(input_dir, image)
        img = cv2.imread(img_path)
        if img is None:
            continue
        video_writer.write(img)
    video_writer.release()
    print(f"Video saved to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extracts video from a ROS bag file or all bag files in a directory.')
    parser.add_argument('--dataset', type=str, default='', help='Name of the dataset to process. [TorWIC, TorWIC-SLAM, SCAND]')
    parser.add_argument('--bag_file', type=str, help='Path to the input ROS bag file.')
    parser.add_argument('--input_dir', type=str, help='Path to a directory containing ROS bag files, or directories with images.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output MP4 files (used with --bag_dir).')
    parser.add_argument('--output_file', type=str, help='Path for the output MP4 video file (used with --bag_file).')
    args = parser.parse_args()

    if args.dataset not in ['TorWIC', 'TorWIC-SLAM', 'SCAND']:
        print("Error: Invalid dataset name. Please use 'TorWIC', 'TorWIC-SLAM', or 'SCAND'.")
        exit(1)
    if args.dataset == 'TorWIC-SLAM':
        if args.input_dir:
            if not os.path.isdir(args.input_dir):
                print(f"Error: The directory '{args.input_dir}' does not exist.")
            else:
                output_dir = args.output_dir or args.input_dir
                os.makedirs(output_dir, exist_ok=True)
                for exp_name in os.listdir(args.input_dir):
                    if not os.path.isdir(os.path.join(args.input_dir, exp_name)):
                        continue
                    for direction in ['image_left', 'image_right']:
                        dir_path = os.path.join(args.input_dir, exp_name, direction)
                        output_file = os.path.join(output_dir, exp_name + '_' + direction + '.mp4')
                        print(f"Processing {exp_name} -> {output_file}")
                        extract_images_to_video(dir_path, output_file)
        else:
            print("Error: You must provide --input_dir to use TorWIC-SLAM.")
    elif args.dataset == 'SCAND' or args.dataset == 'TorWIC':
        if args.input_dir:
            if not os.path.isdir(args.input_dir):
                print(f"Error: The directory '{args.input_dir}' does not exist.")
            else:
                output_dir = args.output_dir or args.input_dir
                os.makedirs(output_dir, exist_ok=True)
                for fname in os.listdir(args.input_dir):
                    if fname.endswith('.bag'):
                        bag_path = os.path.join(args.input_dir, fname)
                        output_file = os.path.join(
                            output_dir, os.path.splitext(fname)[0] + '.mp4')
                        print(f"Processing {bag_path} -> {output_file}")
                        extract_bag_to_video(args.dataset, bag_path, output_file)
        else:
            if not args.bag_file:
                print("Error: You must provide either --bag_file or --input_dir.")
            else:
                output_file = args.output_file
                if not output_file or not output_file.endswith('.mp4'):
                    output_file = os.path.splitext(args.bag_file)[0] + '.mp4'
                extract_bag_to_video(args.dataset, args.bag_file, output_file)
