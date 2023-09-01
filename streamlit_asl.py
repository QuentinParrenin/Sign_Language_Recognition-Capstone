import streamlit as st
import os 
import imageio
import pandas as pd
from PIL import Image

import tensorflow as tf 

st.set_page_config(layout ="wide")

with st.sidebar:
    st.title('Here are the numbers in Sign Language, what if we practice ?')
    st.image("https://signstation.org/wp-content/uploads/2021/09/How-to-Count-in-Sign-Language.jpg")
    st.title("The model is focused to understand numbers, adresses and email for American Sign Language")
    st.text("..................................") 
    st.title('But how ? Using Pre-LN Transformer architecture')

    image_path = "d:\Bureau\ASLCapstone\Pre-LN_transformer.PNG"  # Replace with the actual path to your image

# Open and display the image using PIL
    image = Image.open(image_path)

# Display the image in Streamlit
    st.image(image, caption="", use_column_width=True)

st.title('Sign Language Recognition') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('videosasl'))

selected_video = st.selectbox('Choose video : ', options)

data = {
    'x_right_hand_0': [0.373361, 0.376213, 0.377942, 0.380662, 0.378948],
    'x_right_hand_1': [0.435100, 0.445086, 0.452504, 0.456965, 0.454280],
    'x_right_hand_2': [0.489499, 0.505168, 0.513839, 0.518692, 0.516342],
    'x_right_hand_3': [0.526727, 0.545858, 0.553987, 0.559151, 0.555690],
    'x_right_hand_4': [0.536582, 0.551980, 0.563893, 0.568610, 0.564724],
    'x_right_hand_5': [0.448423, 0.459995, 0.472921, 0.478644, 0.472442],
    'x_right_hand_6': [0.463262, 0.486890, 0.501034, 0.505475, 0.504126],
    'x_right_hand_7': [0.522496, 0.546788, 0.561898, 0.564635, 0.562642],
    'x_right_hand_8': [0.576127, 0.599532, 0.612804, 0.616860, 0.613706],
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)



# # Generate two columns 
col1, col2, col3 = st.columns(3)

if options: 

    with col1:
        st.info('The video below displays the features extracted from the signer. Having Hand, Pose and Face landmarks.')
        file_path = os.path.join('videosasl', selected_video)
        os.system(f'ffmpeg -i {file_path} test_video.mp4')
        video = open(file_path, 'rb')
        video_bytes = video.read()

        st.video(video_bytes)
        st.markdown("### The model is doing specially well in this scenario where the hand is clearly visible, and the signs are relatively slow")
        st.markdown("### The model works well with both left hand and right hand signer")
        st.markdown("### Hand position is definitely essential for the model and pose of the arm can add some extra insights specially when the signer is far enough")


    with col2: 
        st.info("Example of the type of data the model is analyzing (with 21 hand landmarks, 15 body pose total and the x, y and z dimensions):")
        st.dataframe(df.style.hide_index(), width=500, height=200)


        st.info(" The hand position is the most important! Here is how it looks like for the model with xyz.")
        image_path2 = "d:\Bureau\ASLCapstone\hand_image_landmark.PNG"  # Replace with the actual path to your image
        image = Image.open(image_path2)
        st.image(image, caption="", use_column_width=True)


        
    with col3:
        st.markdown('<p style="color: red; font-weight: bold; font-size: 24px; border: 2px solid red; padding: 10px;">The prediction of the model is: 395-245-5739</p>', unsafe_allow_html=True)






        st.title("DID YOU KNOW ?")
        st.markdown("### Worldwide about 72 millions persons use sign language to communicate")
        st.markdown("### Signers have a dominant hand as well! About 12% of signers use left hand")
        st.markdown("### Signers can sign up to 8 characters or numbers a second ")
        st.markdown("#### -> Signing is significantly faster for signers than typing, same as speakers when we talk!")
