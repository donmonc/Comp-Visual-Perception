# mooney_app.py
# Small Gradio app to make Mooney images.
# You can try the settings on one image, then process a whole folder.
#
# Install: pip install gradio opencv-python numpy
# Run:     python mooney_app.py   (opens in the browser)

import os

import cv2
import gradio as gr
import numpy as np

IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def make_mooney(image, blur_size, white_percent):
    # resize so the longer side is 512 (same blur effect for all images)
    height, width = image.shape[:2]
    scale = 512 / max(height, width)
    image = cv2.resize(image, (int(width * scale), int(height * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur kernel has to be odd
    blur_size = int(blur_size)
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # threshold so that white_percent of the pixels are white
    threshold = np.percentile(blurred, 100 - white_percent)
    mooney = np.where(blurred > threshold, 255, 0).astype(np.uint8)
    return mooney


def get_files(folder):
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for filename in sorted(os.listdir(folder)):
        if os.path.splitext(filename)[1].lower() in IMAGE_TYPES:
            files.append(os.path.join(folder, filename))
    return files


def load_folder(folder):
    files = get_files(folder)
    if len(files) == 0:
        return gr.update(maximum=1, value=0), "No images found in this folder."
    return gr.update(maximum=len(files) - 1, value=0), str(len(files)) + " images loaded."


def preview(folder, index, uploaded, blur_size, white_percent):
    # gradio gives uploaded images as RGB, opencv wants BGR
    if uploaded is not None:
        image = cv2.cvtColor(uploaded, cv2.COLOR_RGB2BGR)
        info = "uploaded image"
    else:
        files = get_files(folder)
        if len(files) == 0:
            return None, None, "Upload an image or load a folder."
        path = files[int(index) % len(files)]
        image = cv2.imread(path)
        info = os.path.basename(path)

    result = make_mooney(image, blur_size, white_percent)
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return original, result, info


def process_folder(folder, output_folder, blur_size, white_percent, progress=gr.Progress()):
    files = get_files(folder)
    if len(files) == 0:
        return "No images found in the input folder."

    os.makedirs(output_folder, exist_ok=True)
    for path in progress.tqdm(files):
        image = cv2.imread(path)
        if image is None:
            continue
        result = make_mooney(image, blur_size, white_percent)
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(output_folder, name + ".png"), result)

    return "Done! " + str(len(files)) + " images saved to " + output_folder


with gr.Blocks(title="Mooney generator") as demo:
    gr.Markdown("## Mooney image generator")

    with gr.Row():
        with gr.Column(scale=1):
            folder = gr.Textbox(label="Input folder", value="Original")
            load_button = gr.Button("Load folder")
            index = gr.Slider(0, 1, value=0, step=1, label="Image number")
            uploaded = gr.Image(label="or upload a single image", type="numpy", height=150)

            blur_size = gr.Slider(1, 61, value=15, step=2, label="Blur size (pixels)")
            white_percent = gr.Slider(5, 95, value=40, step=1, label="White pixels (%)")

            output_folder = gr.Textbox(label="Output folder", value="Mooney")
            batch_button = gr.Button("Process whole folder", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            with gr.Row():
                original_view = gr.Image(label="Original")
                mooney_view = gr.Image(label="Mooney")

    inputs = [folder, index, uploaded, blur_size, white_percent]
    outputs = [original_view, mooney_view, status]

    load_button.click(load_folder, folder, [index, status]).then(preview, inputs, outputs)

    # load the default folder as soon as the page opens
    demo.load(load_folder, folder, [index, status]).then(preview, inputs, outputs)

    # update the preview whenever a setting changes
    for control in [index, uploaded, blur_size, white_percent]:
        control.change(preview, inputs, outputs)

    batch_button.click(process_folder,
                       [folder, output_folder, blur_size, white_percent], status)


demo.launch()