import streamlit as st
import numpy as np
from numpy import random
from utils import Discriminator, Generator, plot_image

# Function to display images
# def view_samples(samples, m, n):
#     fig, axes = plt.subplots(figsize=(5, 5), nrows=m, ncols=n, sharey=True, sharex=True)
#     axes = np.array(axes).reshape(m, n)  # Ensure axes is always a 2D array
#     for ax, img in zip(axes.ravel(), samples):
#         ax.xaxis.set_visible(False)
#         ax.yaxis.set_visible(False)
#         ax.imshow(1 - np.array(img).reshape((2,2)), cmap='Greys_r')  
#     st.pyplot(fig)

np.random.seed(42)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# The GAN
D = Discriminator(learning_rate)
G = Generator(learning_rate)

# Streamlit UI
def main():
    st.title("GAN 2 by 2 Pixel Image Generator")
    
    st.subheader("Select Pixel Values")
    col1, col2 = st.columns(2)
    with col1:
        pixel_00 = st.slider("Pixel (0,0)", 0, 255, 128)
        pixel_10 = st.slider("Pixel (1,0)", 0, 255, 128)
    with col2:
        pixel_01 = st.slider("Pixel (0,1)", 0, 255, 128)
        pixel_11 = st.slider("Pixel (1,1)", 0, 255, 128)
    pixels = [pixel_00, pixel_01, pixel_10, pixel_11]
    pixels=[i/255 for i in pixels]
   
    # Examples of faces
    faces = [np.array(pixels),
            np.array(pixels),
            np.array(pixels),
            np.array(pixels),
            np.array(pixels)]
        
    _ = plot_image(np.array(pixels))
    noise = [np.random.randn(2,2) for i in range(20)]
    
    z_value = st.number_input("Enter Z Value", value=0.5, step=0.1)

    # For the error plot
    errors_discriminator = []
    errors_generator = []

    for epoch in range(epochs):
        
        for face in faces:
            
            # Update the discriminator weights from the real face
            D.update_from_image(face)

            z = random.rand()

            # Calculate the errors
            errors_discriminator.append(sum(D.error_from_image(face) + D.error_from_noise(z)))
            errors_generator.append(G.error(z, D))
    
            # Build a fake face
            noise = G.forward(z)
            
            # Update the weights from the fake face
            D.update_from_noise(noise)
            G.update(z, D)
    
    if st.button("Generate Image"):
        generated_image = G.forward(z_value)
        st.subheader("Generated Image")
        plot_image(generated_image)

if __name__ == "__main__":
    main()
