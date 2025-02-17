import cv2
import os
import glob

# Paths to images
original_images_folder = "data/images/"  # Folder with original images
plot_images_folder = "data_flipped/plot_images_2_7/"  # Folder with generated plots
output_video = "video_flipped_2_7.mp4"  # Output video filename

# Get sorted lists of images
original_images = sorted(glob.glob(os.path.join(original_images_folder, "*.jpg")))  # Adjust extension if needed
plot_images = sorted(glob.glob(os.path.join(plot_images_folder, "*.png")))

# Ensure images match
## assert len(original_images) == len(plot_images), "Mismatch in number of original and plot images"

# Read the first image to get dimensions
frame = cv2.imread(original_images[0])
plot = cv2.imread(plot_images[0])

# Resize to match (assuming original images are larger)
plot = cv2.resize(plot, (frame.shape[1], frame.shape[0]))

# Video settings
frame_width = frame.shape[1] * 2  # Double width for side-by-side
frame_height = frame.shape[0]
fps = 20  # Adjust FPS as needed

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Process images and save to video
for orig_path, plot_path in zip(original_images[1:len(original_images) + 1], plot_images):
    orig_frame = cv2.imread(orig_path)
    orig_frame = cv2.flip(orig_frame, 1)
    plot_frame = cv2.imread(plot_path)

    # Resize the plot to match the original frame
    plot_frame = cv2.resize(plot_frame, (orig_frame.shape[1], orig_frame.shape[0]))

    # Concatenate images side by side
    combined_frame = cv2.hconcat([orig_frame, plot_frame])

    # Write to video
    out.write(combined_frame)

# Release video writer
out.release()
print(f"Video saved as {output_video}")