import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import json
from datetime import datetime
import sys
import traceback

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

def progress_callback(batch_idx, step_idx, total_steps, total_batches):
    percentage = int(((batch_idx + step_idx / total_steps) / total_batches) * 100)
    progress = {
        "type": "progress",
        "batch": batch_idx + 1,
        "total_batches": total_batches,
        "step": step_idx + 1,
        "total_steps": total_steps,
        "percentage": percentage
    }
    print(json.dumps(progress), flush=True)

def text_to_image(pipe, prompt, negative_prompt, batch_size, height, width, steps, guidance):
    images = []
    for batch_idx in range(batch_size):
        print(f"[INFO] Generating batch {batch_idx+1}/{batch_size} (text-to-image)...", flush=True)
        generator = torch.Generator(device=DEVICE).manual_seed(torch.randint(0, 2**31, (1,)).item())
        def callback(step, t, latents):
            progress_callback(batch_idx, step, steps, batch_size)
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            callback=callback,
            callback_steps=1
        ).images[0]
        images.append(img)
    return images

def image_to_image(pipe, prompt, negative_prompt, image_path, batch_size, strength, height, width, steps, guidance):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    init_image = Image.open(image_path).convert("RGB")
    if init_image.size != (width, height):
        init_image = init_image.resize((width, height))

    images = []
    for batch_idx in range(batch_size):
        print(f"[INFO] Generating batch {batch_idx+1}/{batch_size} (image-to-image)...", flush=True)
        generator = torch.Generator(device=DEVICE).manual_seed(torch.randint(0, 2**31, (1,)).item())
        def callback(step, t, latents):
            progress_callback(batch_idx, step, steps, batch_size)
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            callback=callback,
            callback_steps=1
        ).images[0]
        images.append(img)
    return images

def run_from_json(json_string):
    try:
        config = json.loads(json_string)
        model_id = config.get("model_id", "runwayml/stable-diffusion-v1-5")
        use_image = config.get("use_image", False)
        prompt = config.get("prompt", "")
        negative_prompt = config.get("negative_prompt", "")
        batch_size = int(config.get("batch_size", 1))
        height = int(config.get("height", 512))
        width = int(config.get("width", 512))
        steps = int(config.get("steps", 20))
        guidance = float(config.get("guidance", 7.5))
        strength = float(config.get("strength", 0.4))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(OUTPUT_ROOT, f"output_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        print(f"[INFO] Output folder: {output_folder}", flush=True)

        print(f"[INFO] Loading model pipelines for {model_id}...", flush=True)
        pipe_t2i = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
        pipe_t2i.safety_checker = None
        pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
        pipe_i2i.safety_checker = None
        print("[INFO] Pipelines loaded successfully.", flush=True)

        images = []
        if use_image:
            input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not input_files:
                raise FileNotFoundError("No input images found in 'input' folder")
            input_path = os.path.join(INPUT_DIR, input_files[0])
            print(f"[INFO] Using input image: {input_path}", flush=True)
            images = image_to_image(pipe_i2i, prompt, negative_prompt, input_path, batch_size, strength, height, width, steps, guidance)
            filenames = []
            for idx, img in enumerate(images):
                path = os.path.join(output_folder, f"img2img_{idx+1}.png")
                img.save(path)
                filenames.append(path)
        else:
            images = text_to_image(pipe_t2i, prompt, negative_prompt, batch_size, height, width, steps, guidance)
            filenames = []
            for idx, img in enumerate(images):
                path = os.path.join(output_folder, f"text2img_{idx+1}.png")
                img.save(path)
                filenames.append(path)

        final_json = {"type": "result", "output_folder": output_folder, "images": filenames}
        print(json.dumps(final_json), flush=True)

    except Exception as e:
        error_json = {"type": "error", "message": str(e), "traceback": traceback.format_exc()}
        print(json.dumps(error_json), flush=True)

if __name__ == "__main__":
    input_json = sys.stdin.read()
    run_from_json(input_json)
