import streamlit as st
import diffractsim
from diffractsim import PolychromaticField, Lens, ApertureFromImage, cf, nm, mm, cm
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import shutil
from PIL import Image
import cv2

# Set backend
diffractsim.set_backend("CPU")  # Change to "CUDA" for GPU acceleration

def images_to_video(image_folder, fps, output_dir):
    img_array = []
    frame_size = None

    # Sort the image files by name
    image_files = sorted(os.listdir(image_folder))

    for file_name in image_files:
        img = cv2.imread(os.path.join(image_folder, file_name))
        if img is None:
            continue

        # Set frame size based on the first image
        if frame_size is None:
            frame_size = (img.shape[1], img.shape[0])

        img = cv2.resize(img, frame_size)
        img_array.append(img)

    # Temporary file to store video
    video_path = os.path.join(output_dir, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for img in img_array:
        out.write(img)

    out.release()
    return video_path

def main():
    st.title("Diffraction Simulation and Video Generation")

    # User input for aperture image
    uploaded_file = st.file_uploader("Choose an image file for aperture", type=["jpg", "png", "jpeg"])

    # Inputs for customizing the linspace range, number of steps, focal length, and axis visibility
    z_start = st.number_input("Start of linspace range (cm)", value=0.0, format="%.5f")
    z_end = st.number_input("End of linspace range (cm)", value=0.11, format="%.5f")
    num_steps = st.slider("Number of steps in linspace", min_value=1, max_value=100, value=60)
    focal_length = st.number_input("Focal length of the lens (cm)", value=0.22, format="%.5f")
    show_axis = st.checkbox("Show axis in plots", value=False)

    # Directory to save images (use a temporary directory in Streamlit)
    output_dir = "temp_images"
    os.makedirs(output_dir, exist_ok=True)

    if st.button("Generate Diffraction Images"):
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

                # Initialize PolychromaticField
                F = PolychromaticField(
                    spectrum=4*cf.illuminant_d65,
                    extent_x=15. * mm, extent_y=15. * mm,
                    Nx=1500, Ny=1500
                )

                # Add ApertureFromImage
                F.add(ApertureFromImage(temp_file_path, image_size=(5. * mm, 5 * mm), simulation=F))

                # Add Lens with customizable focal length
                F.add(Lens(f=focal_length * cm))

                # Prepare for capturing images
                z_values = np.linspace(z_start * cm, z_end * cm, num_steps)  # Use the custom linspace values

                for i, z in enumerate(z_values):
                    F.propagate(z=z)
                    rgb = F.get_colors()

                    # Create and configure the plot
                    plt.figure(figsize=(6, 6))
                    F.plot_colors(rgb, xlim=[-6*mm, 6*mm], ylim=[-6*mm, 6*mm])
                    
                    # Configurable axis visibility
                    if show_axis:
                        plt.axis('on')
                    else:
                        plt.axis('off')

                    # Save image directly to the folder
                    output_path = os.path.join(output_dir, f'image_{i:03d}.png')
                    plt.savefig(output_path, format='png')
                    plt.close()

                    # Display image in Streamlit
                    st.image(output_path)

                st.success("All images saved and displayed successfully.")

    # Zip the images and provide a download button
    if st.button("Download Images as ZIP"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
            with tempfile.TemporaryDirectory() as temp_image_dir:
                # Copy images to temporary directory
                for image_name in os.listdir(output_dir):
                    shutil.copy(os.path.join(output_dir, image_name), temp_image_dir)

                # Create a zip file
                shutil.make_archive(temp_zip.name[:-4], 'zip', temp_image_dir)

            with open(temp_zip.name, "rb") as file:
                st.download_button(label="Download Images and video as ZIP", data=file, file_name="images.zip")

    # Video generation
    fps = st.slider("Frames per second for video", min_value=1, max_value=60, value=30)
    if st.button("Generate Video"):
        if os.listdir(output_dir):
            try:
                video_path = images_to_video(output_dir, fps, output_dir)
                st.success("Video created successfully!")
                with open(video_path, "rb") as file:
                    st.download_button(label='Download Video', data=file, file_name='output_video.mp4', mime='video/mp4')
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("No images found. Please generate images first.")

if __name__ == "__main__":
    main()
