import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import os
import io
import json
import warnings
import logging
from password import check_password
import itertools


# Configuration
width = 700
height = 400
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
try:
    with open('Tokens/tokens.json', 'r') as file:
        tokens = json.load(file)
    stability_key = tokens['STABILITY_TOKEN']
    st.session_state['LOCAL_MACHINE'] = True
except:
    stability_key = st.secrets['STABILITY_TOKEN']
    st.session_state['LOCAL_MACHINE'] = False

# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=stability_key, # Api key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-768-v2-1", # Set the engine to use for generation. 
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0 
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)

st.title('Stable Diffusion Interface')


if (('LOCAL_MACHINE' in st.session_state.keys()) and (st.session_state['LOCAL_MACHINE'])) or (check_password()):
    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox("Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform"))

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FF8000")
    bg_color = "#FFFFFF" # st.sidebar.color_picker("Background color hex: ", "#FFFFFF")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = True # st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        width=width, 
        height=height,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
        display_toolbar=True
    )
    
    schedules = [0.85,0.9,0.95]
    scales = [7.0]
    logging.info(schedules)
    logging.info(scales)
    prompt = st.text_input("")
    button = st.button(label='Generate Image')
    imgs = []
    if button:
        for schedule, scale in itertools.product(schedules, scales):
            # Set up our initial generation parameters.
            answers = stability_api.generate(
                # If you have an init image
                init_image=Image.fromarray(canvas_result.image_data).resize((256,256)),
                prompt=prompt,
                start_schedule=schedule,
                seed=992446758, # If a seed is provided, the resulting generated image will be deterministic.
                                # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                                # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
                steps=30, # Amount of inference steps performed on image generation. Defaults to 30. 
                cfg_scale=scale, # Influences how strongly your generation is guided to match your prompt.
                            # Setting this value higher increases the strength in which it tries to match your prompt.
                            # Defaults to 7.0 if not specified.
                width=512, # Generation width, defaults to 512 if not included.
                height=512, # Generation height, defaults to 512 if not included.
                samples=1, # Number of images to generate, defaults to 1 if not included.
                sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                            # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                            # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
            )

            # Set up our warning to print to the console if the adult content classifier is tripped.
            # If adult content classifier is not tripped, save generated images.
            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        warnings.warn(
                            "Your request activated the API's safety filters and could not be processed."
                            "Please modify the prompt and try again.")
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        img = Image.open(io.BytesIO(artifact.binary))
                        imgs.append(img)
                        # st.image(img, width=width)
                        # img.save(str(artifact.seed)+ ".png") # Save our generated images with their seed number as the filename.
            logging.info(f'PROCESSED SCALE {scale} AND SCHEDULE {schedule}')
            
        divider = 3
        for idx_r in range(len(schedules)):
            cols = st.columns(len(scales))
            for idx_c, col in enumerate(cols):
                col.image(imgs[(idx_r*len(cols))+idx_c].resize((width//divider, height//divider)), width=width//divider)
                        