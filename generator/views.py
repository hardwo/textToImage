import torch
from diffusers import StableDiffusionPipeline
from django.shortcuts import render
from .models import GeneratedImage
from django.core.files.base import ContentFile
import io
from PIL import Image

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

def home(request):
    if request.method == "POST":
        prompt = request.POST.get("prompt")
        image = generate_image(prompt)

        # Save image to model
        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_instance = GeneratedImage(prompt=prompt)
        img_instance.image.save(f"{prompt[:30]}.png", ContentFile(img_io.getvalue()), save=True)

    images = GeneratedImage.objects.all().order_by("-created_at")
    return render(request, "generator/home.html", {"images": images})


# Create your views here.
