# stable-diffusion-test
Example getting stable diffusion models running locally with low-er memory GPUs (8-16 GB)

## Setup
1. Sign in to huggingface: https://huggingface.co/welcome
2. Accept stable diffusion agreement: https://huggingface.co/CompVis/stable-diffusion-v1-4
3. Create an access token and copy it to a file in this repo nammed `.HF_ACCESS_TOKEN`: https://huggingface.co/settings/tokens
4. pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu113
5. Run `txt2img.py` using any prompt you want. e.g:
> python3 txt2img.py Cosmic chicken giving birth to the universe as an egg

See also `stable_diffusion_examples.ipynb` for a notebook version of the script
