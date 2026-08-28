"""
Generates 5 COCO-style captions per THINGS image using Qwen3-VL-8B-Instruct

"""

import torch
import json
import os
import re
import time
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

start_time = time.time()

# --- Configuration & Paths ---
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
IMAGE_DIR = "/scratch/jeffreykatab/Code/Encoding_Models/THINGS/Images_and_Metadata/object_images"
METADATA_PATH = "/scratch/jeffreykatab/Code/Encoding_Models/THINGS/fMRI/betas_csv/sub-01_StimulusMetadata.csv"
SAVE_PATH = "/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/image_descriptions_qwen.json"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True) 
CHECKPOINT_INTERVAL = 100

MAX_NEW_TOKENS = 60
TEMPERATURE = 0.7
TOP_P = 0.9

_STYLE_RULES = (
    "Use a single plain, literal sentence of 20-30 words. Do not use "
    "metaphors, opinions, mood words, speculation, or embellishment. Do "
    "not use quotation marks, titles, or markdown. Do not say \"image\" "
    "or \"photo\"."
)

# --- Facet-varied prompts: each targets a different aspect of the image and
# uses a different sentence-opening style in its few-shot examples, so the
# model isn't locked onto one template across all 5 captions. ---
FACET_PROMPTS = [
    # 1. Main subject / action / state
    (
        "Write one caption for this image describing the main subject or "
        "object and what it is doing or how it is positioned. "
        + _STYLE_RULES + " Match this style:\n"
        "- A young girl in a red jacket pedals a bicycle down a paved "
        "path, leaning slightly forward with both hands on the handlebars.\n"
        "- A brown dog lies stretched out on a grassy lawn, its head "
        "resting on its front paws near a low wooden fence.\n"
        "Now write one caption in this style, focused on the main subject "
        "and its action or position, for the image below."
    ),
    # 2. Spatial layout / background / setting
    (
        "Write one caption for this image describing where things are "
        "positioned relative to each other and what is visible in the "
        "background or surrounding setting. "
        + _STYLE_RULES + " Match this style:\n"
        "- In the background, a low wooden fence runs along the edge of "
        "a grassy lawn beside a narrow paved path.\n"
        "- Behind the two people seated on the bench, a calm lake "
        "stretches toward a row of distant trees.\n"
        "Now write one caption in this style, focused on spatial layout "
        "and background, for the image below."
    ),
    # 3. Colors, materials, textures
    (
        "Write one caption for this image describing the colors, "
        "materials, and textures of the main objects or surfaces visible. "
        + _STYLE_RULES + " Match this style:\n"
        "- The countertop is dark speckled granite, paired with a "
        "brushed steel faucet and a pale ceramic sink beside it.\n"
        "- A red cotton jacket with metal zippers contrasts against the "
        "glossy blue enamel frame of the bicycle underneath it.\n"
        "Now write one caption in this style, focused on colors, "
        "materials, and textures, for the image below."
    ),
    # 4. A distinguishing / unusual detail
    (
        "Write one caption for this image describing one specific, "
        "distinguishing, or unusual detail that stands out -- something "
        "a viewer might otherwise miss. "
        + _STYLE_RULES + " Match this style:\n"
        "- One panel of the wooden fence is visibly cracked and leans "
        "slightly outward near the corner of the lawn.\n"
        "- A small chip is visible along the rim of the ceramic sink, "
        "just beneath the base of the faucet.\n"
        "Now write one caption in this style, focused on a single "
        "distinguishing detail, for the image below."
    ),
    # 5. Counts/quantities or a different sentence structure
    (
        "Write one caption for this image that starts by stating a count "
        "or quantity of visible objects, or otherwise uses a sentence "
        "structure different from a typical 'A [subject] [verb]...' "
        "opening. " + _STYLE_RULES + " Match this style:\n"
        "- Three wooden chairs surround a round glass table set on a "
        "tiled patio bordered by potted plants.\n"
        "- Resting against the fence, a single bicycle with a red frame "
        "faces away from the paved path.\n"
        "Now write one caption in this style, using a count or a "
        "non-standard sentence opening, for the image below."
    ),
]
N_CAPTIONS = len(FACET_PROMPTS)

# --- Cleanup for any stray formatting the model adds despite instructions ---
_LEADING_MARKUP_RE = re.compile(r'^\s*(?:\d+[\.\)]\s*|[-*â€¢]\s*|["\'*])+')
_TRAILING_MARKUP_RE = re.compile(r'["\'*]+\s*$')


def clean_caption(text):
    text = text.strip()
    text = _LEADING_MARKUP_RE.sub('', text)
    text = _TRAILING_MARKUP_RE.sub('', text)
    return text.strip()


# 1) Load Model and Processor
print(f"[Loading] {MODEL_ID} locally...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    # attn_implementation="sdpa"  # Use sdpa if flash-attention installation is problematic
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 2) Data Setup
df = pd.read_csv(METADATA_PATH)
# all_stimuli = df.drop_duplicates(subset=['stimulus']).head(10)
all_stimuli = df.drop_duplicates(subset=['stimulus'])

def generate_one_caption(image, prompt_text, processor, model, previous_captions=None):
    """
    Runs one generate() call with a given facet prompt, returns one cleaned
    caption.
    """
    if previous_captions:
        prompt_text = (
            prompt_text
            + "\n\nCaptions already written for this image (do not repeat "
            "their content or phrasing -- describe different aspects):\n"
            + "\n".join(f"- {c}" for c in previous_captions)
        )

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=1,
        )

    generated_ids_trimmed = generated_ids[:, inputs.input_ids.size(1):]
    decoded = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return clean_caption(decoded)


def get_coco_style_captions(image_path, processor, model):
    """
    Returns a list of N_CAPTIONS cleaned caption strings for one image via
    5 sequential (chained) generate() calls: each uses its own facet prompt
    from FACET_PROMPTS (targeting a different aspect of the image) AND sees
    every caption generated so far for this image, with an explicit
    instruction not to repeat them. 
    """
    image = Image.open(image_path).convert("RGB")
    captions = []
    for prompt in FACET_PROMPTS:
        caption = generate_one_caption(image, prompt, processor, model, previous_captions=captions)
        captions.append(caption)
    return captions


# 3) Processing Loop with Checkpointing and Resume Support
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r") as f:
        descriptions = json.load(f)
    print(f"[Resuming] Loaded {len(descriptions)} already-captioned images from {SAVE_PATH}")
else:
    descriptions = {}

print("[Generating] COCO-style, facet-varied captions with periodic checkpointing...")

n_processed_this_run = 0
for i, row in tqdm(enumerate(all_stimuli.iterrows()), total=len(all_stimuli)):
    _, row_data = row
    filename = row_data['stimulus']

    if filename in descriptions:
        continue  # already captioned in a previous run

    path = os.path.join(IMAGE_DIR, row_data['concept'], filename)

    try:
        descriptions[filename] = get_coco_style_captions(path, processor, model)
    except Exception as e:
        print(f"\nError on {filename}: {e}")

    n_processed_this_run += 1

    if n_processed_this_run % CHECKPOINT_INTERVAL == 0:
        with open(SAVE_PATH, "w") as f:
            json.dump(descriptions, f, indent=4)
        print(f"\n[Checkpoint] Saved at {len(descriptions)} total images captioned.")

# Final save
with open(SAVE_PATH, "w") as f:
    json.dump(descriptions, f, indent=4)

print(f"\n[Done] Process complete. Final file saved to {SAVE_PATH}")

print(f"\nExecution complete! Total time: {time.time() - start_time:.2f} seconds.")