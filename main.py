import streamlit as st
import replicate
import os
from streamlit_image_comparison import image_comparison

# Set your Replicate API token
REPLICATE_API_TOKEN = "r8_N0tX8bUF1lSOh09ShwqZ0rIztKzYpgR3iE9XL"  # Replace with your actual Replicate API token
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# Streamlit app
st.title("High-Resolution ControlNet Tile Transformer")

# Input for external image link
image_url = st.text_input("Enter the external image URL", placeholder="https://example.com/your-image.jpg")

# Display the original image
if image_url:
    st.image(image_url, caption="Original Image", use_column_width=True)

# Prompt input
prompt = st.text_area("Enter your design prompt", placeholder="Describe your desired image transformation...")

# Optional parameters
hdr = st.selectbox("HDR", options=[0, 1], index=0)
steps = st.slider("Steps", min_value=1, max_value=100, value=20, step=1)
scheduler = st.selectbox("Scheduler", options=["DDIM", "PLMS", "LMSD"], index=0)
creativity = st.slider("Creativity", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
guess_mode = st.selectbox("Guess Mode", options=[True, False], index=1)
resolution = st.slider("Resolution", min_value=512, max_value=4096, value=2048, step=256)
resemblance = st.slider("Resemblance", min_value=0.0, max_value=1.0, value=0.99, step=0.01)
guidance_scale = st.slider("Guidance Scale", min_value=0, max_value=20, value=5, step=1)
negative_prompt = st.text_area("Negative Prompt (optional)", placeholder="e.g., Teeth, tooth, longbody, lowres, etc.")
lora_details_strength = st.slider("Lora Details Strength", min_value=0.0, max_value=2.0, value=0.75, step=0.05)
lora_sharpness_strength = st.slider("Lora Sharpness Strength", min_value=0.0, max_value=2.0, value=1.25, step=0.05)

# Generate Design
if st.button("Generate Design"):
    if image_url and prompt:
        try:
            # Run the model
            output = replicate.run(
                "batouresearch/high-resolution-controlnet-tile:8e6a54d7b2848c48dc741a109d3fb0ea2a7f554eb4becd39a25cc532536ea975",
                input={
                    "hdr": hdr,
                    "image": image_url,
                    "steps": steps,
                    "prompt": prompt,
                    "scheduler": scheduler,
                    "creativity": creativity,
                    "guess_mode": guess_mode,
                    "resolution": resolution,
                    "resemblance": resemblance,
                    "guidance_scale": guidance_scale,
                    "negative_prompt": negative_prompt,
                    "lora_details_strength": lora_details_strength,
                    "lora_sharpness_strength": lora_sharpness_strength,
                }
            )

            # Display the image comparison slider
            st.write("## Compare the Original and Transformed Image")
            image_comparison(
                img1=image_url,  # Original image
                img2=output,  # Transformed image
                label1="Original Image",
                label2="Transformed Image",
                width=700  # Adjust width as necessary
            )

        except replicate.exceptions.ReplicateError as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please enter an image URL and a prompt.")
