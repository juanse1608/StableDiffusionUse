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
import requests
from password import check_password
from functions import st_make_grid
import itertools
import requests
from io import BytesIO

st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

# Configuration
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
try:
    with open('Tokens/tokens.json', 'r') as file:
        tokens = json.load(file)
    stability_key = tokens['STABILITY_TOKEN']
    stability_api_key = tokens['STABILITY_API_TOKEN']
    st.session_state['LOCAL_MACHINE'] = True
except:
    stability_key = st.secrets['STABILITY_TOKEN']
    st.session_state['LOCAL_MACHINE'] = False

# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=stability_key, # Api key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-v1-5", # Set the engine to use for generation. 
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0 
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)

st.title('MELI CONTEST - AI ART WITH STABLE DIFFUSION')
st.header('INFORMATION')

if (('LOCAL_MACHINE' in st.session_state.keys()) and (st.session_state['LOCAL_MACHINE'])) or (check_password()):
    # Specify canvas parameters in application
    epsilon = 1e-5
    width = 700
    height = 512
    drawing_mode = 'freedraw'
    stroke_width = 10

    st.selectbox(label='PICK ONE CHOICE TO GENERATE YOUR IMAGE', options=[
        'NO INITIAL IMAGE',
        'INITIAL IMAGE', 
        # 'DRAWABLE CANVAS'
    ], index=0,key='INIT_IMAGE_STATUS') 
    
    init_img = None
    if st.session_state['INIT_IMAGE_STATUS']=='DRAWABLE CANVAS':
        col1, col2 = st.columns(2)

        with col1:
            stroke_color = st.color_picker("STROKE COLOR", "#000000")
        with col2:
            bg_color = st.color_picker("BACKGROUND COLOR", "#FFFFFF")
        bg_image = st.file_uploader("BACKGROUND IMAGE", type=["png", "jpg"])
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
            point_display_radius= 0,
            key="canvas",
            display_toolbar=True
        )
        init_img = Image.fromarray(canvas_result.image_data).resize((512,512))
    elif st.session_state['INIT_IMAGE_STATUS']=='INITIAL IMAGE':
        # st.file_uploader("INITAL IMAGE", type=["png", "jpg"], key='INITIAL_IMAGE')
        st.text_input("WRITE THE URL OF THE IMAGE YOU WANT TO USE", key='INITIAL_IMAGE')
        # init_img = st.session_state['INITIAL_IMAGE'].resize((512,512)) if st.session_state['INITIAL_IMAGE'] is not None else None
        init_img = st.session_state['INITIAL_IMAGE'] if st.session_state['INITIAL_IMAGE'] is not None else None
        
        try:
            init_img_ = requests.get(init_img)
            init_img_ = Image.open(io.BytesIO(init_img_.content)).resize((512,512))
        except:
            init_img_ = None
    
    with st.sidebar:
        st.slider(label='SCHEDULE', help='THE HIGHER THE VALUE, THE HIGHER THE STRONG OF THE INITIAL IMAGE' ,min_value=0.0, max_value=1.0, value=0.5, key='SCHEDULE', step=0.1, disabled=True)
        st.slider(label='SCALE', help='THE HIGHER THE VALUE, THE HIGHER THE STRONG OF THE PROMPT' ,min_value=6, max_value=12, value=7, key='SCALE', disabled=True)
        st.slider(label='SCALE', help='AMOUNT OF INFERENCE STEPS PERFORMED ON IMAGE GENERATION' ,min_value=10, max_value=100, value=30, key='STEPS', step=10, disabled=True)
        st.slider(label='SAMPLES', help='NUMBER OF SAMPLE TO GENERATE' ,min_value=2, max_value=8, value=2, key='SAMPLES', step=2, disabled=True)
    st.text_input("WRITE YOUR PROMPT AND TRY TO PRODUCE THE MOST CREATIVE IMAGES WITH STABLE DIFFUSION", key='PROMPT')
    st.button(label='GENERATE IMAGE', key='GEN_IMG_BUT')
    imgs = []

    if st.session_state['GEN_IMG_BUT']:
        st.header('RESULTS')
        # Set up our initial generation parameters.
        
        if (st.session_state['INIT_IMAGE_STATUS']=='NO INITIAL IMAGE') and (st.session_state['PROMPT'] is not None) and (st.session_state['PROMPT']!=''):
            answers = stability_api.generate(
                init_image=init_img, # Init Image
                prompt=st.session_state['PROMPT'], # Prompt
                start_schedule=st.session_state['SCHEDULE'], # [0,1] closes to one matches init_image
                seed=1608, # Seed
                steps=st.session_state['STEPS'], # Amount of inference steps performed on image generation. Defaults to 30. 
                cfg_scale=st.session_state['SCALE'], # Influences how strongly your generation is guided to match your prompt.
                width=512, # Generation width, defaults to 512 if not included.
                height=512, # Generation height, defaults to 512 if not included.
                samples=st.session_state['SAMPLES'], # Number of images to generate, defaults to 1 if not included.
                sampler=generation.SAMPLER_K_DPMPP_2S_ANCESTRAL, # Sampler
                guidance_preset=generation.GUIDANCE_PRESET_FAST_GREEN # Enables CLIP Guidance.
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
            logging.info(f'PROCESSED IMAGES!')

            divider = len(imgs)
            cols = st.columns(divider)
            for idx in range(divider):
                with cols[idx]:
                    st.image(imgs[idx], width=width//divider)
        elif (st.session_state['PROMPT'] is not None) and (st.session_state['PROMPT']!='') and (init_img_ is not None):
            url = "https://stablediffusionapi.com/api/v5/controlnet"

            payload = json.dumps({
            "key": stability_api_key,
            "model_id": "midjourney",
            "controlnet_model": "canny",
            "auto_hint": "yes",
            "prompt": st.session_state['PROMPT'],
            "negative_prompt": None,
            "init_image": init_img,
            "width": "512",
            "height": "512",
            "samples": str(st.session_state['SAMPLES']),
            "num_inference_steps": "50",
            "safety_checker": "no",
            "enhance_prompt": "no",
            "scheduler": "UniPCMultistepScheduler",
            "guidance_scale": 15.0,
            "strength": 0.7,
            "seed": 1608,
            "webhook": None,
            "track_id": None
            })
            headers = {
            'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)
            response = json.loads(response.text)
            
            imgs = []
            for url_img in response["output"]:
                response_img = requests.get(url_img)
                img = Image.open(io.BytesIO(response_img.content))
                imgs.append(img)
            logging.info(f'PROCESSED IMAGES!')

            rows = len(imgs)

            for idx in range(rows):
                base_img, diffused_img = st.columns(2)
                with base_img:
                    st.image(init_img_, width=width//2)
                with diffused_img:
                    st.image(imgs[idx], width=width//2)

        st.header('SUGGESTED HASHTAG')
        st.info(f'''#AI-DIFFUSION-DATA-CONTEST - MY PROMPT: "{st.session_state["PROMPT"].upper()}"''')

                            