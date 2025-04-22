import cv2
import os
import glob


def img_to_video(traffic_value, desire_value, flipped):
    # Paths to images
    original_images_folder = "data/images/"  # Folder with original images
    if flipped:
        # plot_images_folder = f"data_flipped/plot_images_{traffic_value}_{desire_value}/"
        output_video = f"video_flipped_{traffic_value}_{desire_value}.mp4"
        velocity_images_folder = f"data/velocity/"
        position_images_folder = f"data/position/"
    else:
        # plot_images_folder = f"data/plot_images_{traffic_value}_{desire_value}/"
        output_video = f"video_{traffic_value}_{desire_value}.mp4"
        velocity_images_folder = f"data/velocity/"
        position_images_folder = f"data/position/"

    # Get sorted lists of images
    original_images = sorted(glob.glob(os.path.join(original_images_folder, "*.jpg")))  # Adjust extension if needed
    # plot_images = sorted(glob.glob(os.path.join(plot_images_folder, "*.png")))
    velocity_images = sorted(glob.glob(os.path.join(velocity_images_folder, "*.png")))
    position_images = sorted(glob.glob(os.path.join(position_images_folder, "*.png")))

    # Ensure images match
    ## assert len(original_images) == len(plot_images), "Mismatch in number of original and plot images"

    # Read the first image to get dimensions
    frame = cv2.imread(original_images[0])
    plot_1 = cv2.imread(velocity_images[0])
    plot_2 = cv2.imread(position_images[0])
    # plot = cv2.imread(plot_images[0])

    # Resize to match (assuming original images are larger)
    # plot = cv2.resize(plot, (frame.shape[1], frame.shape[0]))
    plot_1 = cv2.resize(plot_1, (frame.shape[1], frame.shape[0]))
    plot_2 = cv2.resize(plot_2, (frame.shape[1], frame.shape[0]))

    # Video settings
    frame_width = frame.shape[1] * 3  # Double width for side-by-side
    frame_height = frame.shape[0]
    fps = 20  # Adjust FPS as needed

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Process images and save to video
    for orig_path, velocity_path, position_path in zip(original_images[1:len(original_images) + 1], velocity_images, position_images):
        orig_frame = cv2.imread(orig_path)
        if flipped:
            orig_frame = cv2.flip(orig_frame, 1)
        # plot_frame = cv2.imread(plot_path)
        velocity_frame = cv2.imread(velocity_path)
        position_frame = cv2.imread(position_path)

        # Resize the plot to match the original frame
        # plot_frame = cv2.resize(plot_frame, (orig_frame.shape[1], orig_frame.shape[0]))
        velocity_frame = cv2.resize(velocity_frame, (orig_frame.shape[1], orig_frame.shape[0]))
        position_frame = cv2.resize(position_frame, (orig_frame.shape[1], orig_frame.shape[0]))
        # Concatenate images side by side
        combined_frame = cv2.hconcat([orig_frame, velocity_frame, position_frame])

        # Write to video
        out.write(combined_frame)

    # Release video writer
    out.release()
    print(f"Video saved as {output_video}")

if __name__ == '__main__':
    img_to_video(0, 0, False)