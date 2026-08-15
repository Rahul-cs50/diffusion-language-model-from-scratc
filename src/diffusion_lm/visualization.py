import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_mono_font(size=20):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text_to_width(text, max_chars=90):
    out = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.rstrip()
        if not paragraph:
            out.append("")
            continue
        while len(paragraph) > max_chars:
            out.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        out.append(paragraph)
    return out


def make_chat_lines(user_msg, assistant_text, step, total):
    header = "============================== multi-turn chat mode ==============================="
    lines = [header, "<Starting a new chat. Type your message.>", f"(diffusion step {step:03d}/{total:03d})", "", "[You]:", user_msg, "", "[Assistant]:"]
    assistant_text = assistant_text.replace("<|assistant|>", "", 1).replace("<|end|>", "").strip()
    lines.extend(wrap_text_to_width(assistant_text))
    return lines


def render_terminal_frame(lines, width=1200, height=700, font_size=20, margin=20, line_spacing=6):
    img = Image.new("RGB", (width, height), (10, 10, 10))
    draw = ImageDraw.Draw(img)
    font = get_mono_font(font_size)
    y = margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill=(230, 230, 230))
        y += font_size + line_spacing
        if y > height - margin:
            break
    return img


def make_inference_gif(frames, user_prompt, output_path):
    images = []
    total = len(frames)
    for step, decoded in frames:
        images.append(np.array(render_terminal_frame(make_chat_lines(user_prompt, decoded, step, total))))
    imageio.mimsave(output_path, images, duration=0.08)
