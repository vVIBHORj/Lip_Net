# # Import all of the dependencies
# import streamlit as st
# import os 
# import imageio 
# import numpy as np

# import tensorflow as tf 
# from utils import load_data, num_to_char
# from modelutil import load_model

# # Set the layout to the streamlit app as wide 
# st.set_page_config(layout='wide')

# # Setup the sidebar
# with st.sidebar: 
#     st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
#     st.title('LipBuddy')
#     st.info('This application is originally developed from the LipNet deep learning model.')

# st.title('LipNet Full Stack App') 
# # Generating a list of options or videos 
# options = os.listdir(os.path.join('..', 'data', 's1'))
# selected_video = st.selectbox('Choose video', options)

# # Generate two columns 
# col1, col2 = st.columns(2)

# if options: 

#     # Rendering the video 
#     with col1: 
#         st.info('The video below displays the converted video in mp4 format')
#         file_path = os.path.join('..','data','s1', selected_video)
#         os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

#         # Rendering inside of the app
#         video = open('test_video.mp4', 'rb') 
#         video_bytes = video.read() 
#         st.video(video_bytes)
        
#         # video_file = open('test_video.mp4', 'rb') 
#         # video_bytes = video_file.read() 
#         # st.video(video_bytes)



#     with col2: 
#         st.info('This is all the machine learning model sees when making a prediction')
#         #video, annotations = load_data(tf.convert_to_tensor(file_path))
#         #video_frames = (video.numpy() * 255).astype(np.uint8)  # Scale float values to 0–255 and convert to uint8
#         #imageio.mimsave('animation.gif', video_frames, fps=10)
#         #video_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]  # Example

#         video_np = video.numpy()  # shape: (75, 46, 140, 1)
#         video_frames = [(frame.squeeze() * 255).astype(np.uint8) for frame in video_np]
#         video_frames_rgb = [np.stack([f]*3, axis=-1) for f in video_frames]  # Grayscale to RGB
#         imageio.mimsave('animation.gif', video_frames_rgb, fps=10)
#         st.image('animation.gif', width=400)

#         # Ensure frames are in the correct data type and shape
#         #video_frames = [frame.astype(np.uint8) for frame in video_frames]

#         # Save the frames as a GIF
#         #imageio.mimsave('animation.gif', video_frames, fps=10)
#         #st.image('animation.gif', width=400) 

#         st.info('This is the output of the machine learning model as tokens')
#         model = load_model()
#         yhat = model.predict(tf.expand_dims(video, axis=0))
#         decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
#         #st.text(decoder)
        
#         decoder = decoder[0]  # assuming decoder has shape (1, T)
#         char_tokens = num_to_char(tf.convert_to_tensor(decoder))
#         converted_prediction = tf.strings.reduce_join(char_tokens).numpy().decode('utf-8')
#         st.text("Predicted Sentence:")
#         st.success(converted_prediction)


#         # Convert prediction to text
#         st.info('Decode the raw tokens into words')
#         converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
#         st.text(converted_prediction)
        
# Import all of the dependencies##########################################################
# import streamlit as st
# import os 
# import imageio 
# import numpy as np
# import tensorflow as tf 
# from utils import load_data, num_to_char
# from modelutil import load_model

# # Set the layout to the streamlit app as wide 
# st.set_page_config(layout='wide')

# # Setup the sidebar
# with st.sidebar: 
#     st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
#     st.title('LipBuddy')
#     st.info('This application is originally developed from the LipNet deep learning model.')

# st.title('LipNet Full Stack App') 

# # Generating a list of options or videos 
# options = os.listdir(os.path.join('..', 'data', 's1'))
# selected_video = st.selectbox('Choose video', options)

# # Generate two columns 
# col1, col2 = st.columns(2)

# if options:
#     # Paths
#     file_path = os.path.join('..','data','s1', selected_video)

#     # col1 - Display actual video
#     with col1: 
#         st.info('The video below displays the converted video in mp4 format')
#         os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
#         video_file = open('test_video.mp4', 'rb') 
#         video_bytes = video_file.read() 
#         st.video(video_bytes)

#     # col2 - Display processed frames and prediction
#     with col2: 
#         st.info('This is all the machine learning model sees when making a prediction')
        
#         # Load preprocessed video tensor
#         video_tensor, annotations = load_data(tf.convert_to_tensor(file_path))
#         video_np = video_tensor.numpy()  # shape: (75, 46, 140, 1)

#         # Convert grayscale to RGB for GIF
#         video_frames = [(frame.squeeze() * 255).astype(np.uint8) for frame in video_np]
#         video_frames_rgb = [np.stack([f]*3, axis=-1) for f in video_frames]
#         imageio.mimsave('animation.gif', video_frames_rgb, fps=10)
#         st.image('animation.gif', width=400)

#         # Run the model prediction
#         st.info('This is the output of the machine learning model as tokens')
#         model = load_model()
#         yhat = model.predict(tf.expand_dims(video_tensor, axis=0))
#         decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

#         # Decode to characters
#         decoder = decoder[0]  # shape: (T,)
#         char_tokens = num_to_char(tf.convert_to_tensor(decoder))
#         converted_prediction = tf.strings.reduce_join(char_tokens).numpy().decode('utf-8')

#         st.text("Predicted Sentence:")
#         st.success(converted_prediction)
#####################################################################
# Import all of the dependencies
# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('A LipNet Project')
    st.info('Where silence speaks volumes. Discover the power of A Silent Voice.')

st.title('A Silent Voice') 

# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options:
    # Paths
    file_path = os.path.join('..','data','s1', selected_video)

    # col1 - Display actual video
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        video_file = open('test_video.mp4', 'rb') 
        video_bytes = video_file.read() 
        st.video(video_bytes)

    # col2 - Display processed frames and prediction
    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        
        # Load preprocessed video tensor
        video_tensor, annotations = load_data(tf.convert_to_tensor(file_path))
        video_np = video_tensor.numpy()  # shape: (75, 46, 140, 1)

        # Convert grayscale to RGB for GIF
        video_frames = [(frame.squeeze() * 255).astype(np.uint8) for frame in video_np]
        video_frames_rgb = [np.stack([f]*3, axis=-1) for f in video_frames]
        imageio.mimsave('animation.gif', video_frames_rgb, fps=10)
        st.image('animation.gif', width=400)

        # Run the model prediction
        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video_tensor, axis=0))

        # Decode the output using CTC decoding
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

        # Debugging: print the raw decoder output
        st.text("Raw decoder output:")
        st.text(decoder)

        # Filter out the padding tokens (-1) and any invalid tokens (e.g., zeros or empty tokens)
        decoder = decoder[0]  # Assuming decoder has shape (T,)
        decoder = [token for token in decoder if token != -1 and token != 0]  # Remove padding and zero tokens

        # Decode tokens to characters using num_to_char
        char_tokens = num_to_char(tf.convert_to_tensor(decoder))
        converted_prediction = tf.strings.reduce_join(char_tokens).numpy().decode('utf-8')

        # Display the final predicted sentence
        st.text("Predicted Sentence:")
        st.success(converted_prediction)
