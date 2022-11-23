import os
import sys

import pandas as pd
import numpy as np
import random
import time
from datetime import date
from ast import literal_eval

import streamlit as st
import pdfkit
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from streamlit.components.v1 import iframe
import re
from PIL import Image

import pickle
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torchtext.legacy as torchtext
from torch.utils.data import DataLoader
import layers
import sampler as sampler_module
import evaluation
from ranger import Ranger
from scipy import spatial
from sklearn.neighbors import NearestNeighbors

#### ML ####
#@st.cache() 
def runs(params):
    start = time.time()

    ## Background Removal 
    roi, url = Removal.run(params)
    print(" Final processing time of one image : ", time.time() - start)
    start = time.time()
    ## Category detection and Color Recognition -> Style classification
    response, nam_name = YoloV5.run_detection(roi, url)

    print(" Final processing time of one image : ", time.time() - start)
    return response, nam_name

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img



if __name__ == '__main__':
    
    st.set_page_config(layout="wide", page_icon="\U0001F45A", page_title="Wearly Demo page")
    st.title("\U0001F455 Wearly Demo page \U0001F45A")
    '----------------------------------------------'
    st.sidebar.title('MENU')
    add_selectbox1 = st.sidebar.selectbox("Machine Learning",
                                          ("Please choose", "Detection & Recommendation"))
    
    col1, col2, col3 = st.columns([6, 6, 6])
    with col1:
        st.write("")
    with col2:
        uploaded_file = './imgs/wearly.jpg'
        image = Image.open(uploaded_file)
        st.image(image)
    with col3:
        st.write("")

    ### Center Image
    col1, col2, col3, col4 = st.columns([3,6,6,3])
    with col1:
        st.write("")
    with col2:
        uploaded_file = './imgs/sample1.png'
        image = Image.open(uploaded_file)
        st.image(image)
    with col3:
        uploaded_file = './imgs/sample3.png'
        image = Image.open(uploaded_file)
        st.image(image)
    with col4:
        st.write("")

    ### Bottom
    col1, col2, col3 = st.columns([2,8,2])
    with col1:
        st.write("")
    with col2:
        st.markdown("Wearly trend report provides fashion trends predictions analyzed by machine learning algorithms."
             "Trained style classifier can classify fashion images into multi-label, and based on this,"
             "we predict trends in fashion styles over the past 20 years.In addition, it analyzes and provides detailed "
             "information on specific items through fashion item detection and color recognition algorithms.")
    with col3:
        st.write("")
    
    
    '----------------------------------------------'
    if add_selectbox1 == 'Detection & Recommendation':
        
        left, right = st.columns(2)

        right.write("Welcome to the Weary PPOC APP demo page.")
        right.image("imgs/wearly.jpg", width=300)
        right.image("imgs/ppoc.jpg", width=300)

        left.write("Fill in the data:")
        form = left.form("template_form")
        new_title = \
        '<p style="font-family:sans-serif; color:Red; font-size: 15px;">◆ Upload An Fashion Image</p>'
        form.markdown(new_title, unsafe_allow_html=True)
        
        image_file = form.file_uploader("※ Please upload only one fashion image !",type=['png','jpeg','jpg'], accept_multiple_files=False)
        
        new_title = \
        '<p style="font-family:sans-serif; color:Red; font-size: 15px;">◆ Please press the button below</p>'
        form.markdown(new_title, unsafe_allow_html=True)
        
        submit = form.form_submit_button("Getting Started with Processes")
        
        
        
        if image_file is not None:
            img = load_image(image_file)
            st.image(img,width=250)
            with open(os.path.join("tempDir",image_file.name),"wb") as f:
                f.write(image_file.getbuffer())
            

            with st.spinner('Wait for it...'):
                start = time.time()
                detect_result, nam_name = runs(params=image_file.name)
                end = time.time()
                detect_result = literal_eval(detect_result)
            st.success(f'The detected images are as follows: (Running Time  {end - start:.5f} sec)')

            
            u_nn = image_file.name[:-4]
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)   
            try:
                with col1:
                    imm_nn = nam_name[0].split("_")[-1][:-4]
                    nam_nam = nam_name[0]
                    uploaded_file = f'/mnt/hdd1/wearly/streamlit/results/{u_nn}/{nam_nam}'
                    image = Image.open(uploaded_file)
                    image = image.resize((200, 200), Image.ANTIALIAS)
                    st.image(image, width=200)

                    st.metric("Categories", imm_nn)
                    st.metric("Style", detect_result["Img"]["detectionData"][imm_nn]["style"]["main"])
                    color = st.color_picker('Color', detect_result["Img"]["detectionData"][imm_nn]["hexColor"][0])
            except:
                pass

            try:
                with col3:
                    imm_nn = nam_name[1].split("_")[-1][:-4]
                    nam_nam = nam_name[1]
                    uploaded_file = f'/mnt/hdd1/wearly/streamlit/results/{u_nn}/{nam_nam}'
                    image = Image.open(uploaded_file)
                    image = image.resize((200, 200), Image.ANTIALIAS)
                    st.image(image, width=200)
                    
                    st.metric("Categories", imm_nn)
                    st.metric("Style", detect_result["Img"]["detectionData"][imm_nn]["style"]["main"])
                    color = st.color_picker('Color', detect_result["Img"]["detectionData"][imm_nn]["hexColor"][0])
            except:
                pass

            try:
                with col5:
                    imm_nn = nam_name[2].split("_")[-1][:-4]
                    nam_nam = nam_name[2]
                    uploaded_file = f'/mnt/hdd1/wearly/streamlit/results/{u_nn}/{nam_nam}'
                    image = Image.open(uploaded_file)
                    st.image(image, width=200)
                    
                    st.metric("Categories", imm_nn)
                    st.metric("Style", detect_result["Img"]["detectionData"][imm_nn]["style"]["main"])
                    color = st.color_picker('Color', detect_result["Img"]["detectionData"][imm_nn]["hexColor"][0])
            except:
                pass

            try:
                with col7:
                    imm_nn = nam_name[3].split("_")[-1][:-4]
                    nam_nam = nam_name[3]
                    uploaded_file = f'/mnt/hdd1/wearly/streamlit/results/{u_nn}/{nam_nam}'
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"{imm_nn}")
                    
                    st.metric("Categories", imm_nn)
                    st.metric("Style", detect_result["Img"]["detectionData"][imm_nn]["style"]["main"])
                    color = st.color_picker('Color', detect_result["Img"]["detectionData"][imm_nn]["hexColor"][0])
            except:
                pass



            nam_name2 = [it.split("_")[-1][:-4]for it in nam_name]
            nam_name2.insert(0, "Please select an option")
            nam_name.insert(0, "Please select an option")

            tup_nam = tuple(nam_name2)

            '----------------------------------------------'
            new_title = \
            '<p style="font-family:sans-serif; color:Green; font-size: 30px;">Recommendation for similar products</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            option = st.selectbox('Similarity-based recommendation', tup_nam)
            st.write('You selected:', option)

            with st.spinner('Wait for it...'):
                #try:
                if option == tup_nam[1]:
                    nn =  nam_name[1]
                    pred_group = run(f"/mnt/hdd1/wearly/streamlit/results/{u_nn}/{nn}")
                    st.success("The recommended products for this item are as follows.")

                    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
                    #try:
                    with col1:
                        uploaded_file = pred_group[0]
                        image = Image.open(uploaded_file)
                        image = image.resize((200, 200), Image.ANTIALIAS)
                        st.image(image, width=200)
                    #except:
                    #    pass

                    try:
                        with col3:
                            uploaded_file = pred_group[1]
                            image = Image.open(uploaded_file)
                            image = image.resize((200, 200), Image.ANTIALIAS)
                            st.image(image, width=200)
                    except:
                        pass

                    try:
                        with col5:
                            uploaded_file = pred_group[2]
                            image = Image.open(uploaded_file)
                            image = image.resize((200, 200), Image.ANTIALIAS)
                            st.image(image, width=200)
                    except:
                        pass

                    try:
                        with col7:
                            uploaded_file = pred_group[3]
                            image = Image.open(uploaded_file)
                            image = image.resize((200, 200), Image.ANTIALIAS)
                            st.image(image, width=200)
                    except:
                        pass

                    try:
                        with col9:
                            uploaded_file = pred_group[4]
                            image = Image.open(uploaded_file)
                            image = image.resize((200, 200), Image.ANTIALIAS)
                            st.image(image, width=200)
                    except:
                        pass

                #except:
                #    pass

                try:
                    if option == tup_nam[2]:
                        nn =  nam_name[2]
                        pred_group = run(f"/mnt/hdd1/wearly/streamlit/results/{u_nn}/{nn}")
                        st.success("The recommended products for this item are as follows.")
                        
                        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
                        try:
                            with col1:
                                uploaded_file = pred_group[0]
                                image = Image.open(uploaded_file)
                                image = image.resize((200, 200), Image.ANTIALIAS)
                                st.image(image, width=200)
                        except:
                            pass
                        
                        try:
                            with col3:
                                uploaded_file = pred_group[1]
                                image = Image.open(uploaded_file)
                                image = image.resize((200, 200), Image.ANTIALIAS)
                                st.image(image, width=200)
                        except:
                            pass
                        
                        try:
                            with col5:
                                uploaded_file = pred_group[2]
                                image = Image.open(uploaded_file)
                                image = image.resize((200, 200), Image.ANTIALIAS)
                                st.image(image, width=200)
                        except:
                            pass
                        
                        try:
                            with col7:
                                uploaded_file = pred_group[3]
                                image = Image.open(uploaded_file)
                                image = image.resize((200, 200), Image.ANTIALIAS)
                                st.image(image, width=200)
                        except:
                            pass
                        
                        try:
                            with col9:
                                uploaded_file = pred_group[4]
                                image = Image.open(uploaded_file)
                                image = image.resize((200, 200), Image.ANTIALIAS)
                                st.image(image, width=200)
                        except:
                            pass
                        
                except:
                    pass

                try:
                    if option == tup_nam[3]:
                        nn =  nam_name[3]
                        pred_group = run(f"/mnt/hdd1/wearly/streamlit/results/{u_nn}/{nn}")
                        st.success("The recommended products for this item are as follows.")
                        
                        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
                        try:
                            with col1:
                                uploaded_file = pred_group[0]
                                image = Image.open(uploaded_file)
                                image = image.resize((200, 200), Image.ANTIALIAS)
                                st.image(image, width=200)
                        except:
                            pass
                        
                        try:
                            with col3:
                                uploaded_file = pred_group[1]
                                image = Image.open(uploaded_file)
                                image = image.resize((200, 200), Image.ANTIALIAS)
                                st.image(image, width=200)
                        except:
                            pass
                        
                        try:
                            with col5:
                                uploaded_file = pred_group[2]
                                image = Image.open(uploaded_file)
                                image = image.resize((200, 200), Image.ANTIALIAS)
                                st.image(image, width=200)
                        except:
                            pass
                        
                        try:
                            with col7:
                                uploaded_file = pred_group[3]
                                image = Image.open(uploaded_file)
                                image = image.resize((200, 200), Image.ANTIALIAS)
                                st.image(image, width=200)
                        except:
                            pass
                        
                        try:
                            with col9:
                                uploaded_file = pred_group[4]
                                image = Image.open(uploaded_file)
                                image = image.resize((200, 200), Image.ANTIALIAS)
                                st.image(image, width=200)
                        except:
                            pass
                        
                except:
                    pass

                
                
