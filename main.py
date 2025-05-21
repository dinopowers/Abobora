from diffusers import OnnxStableDiffusionPipeline
height=512
width=512
num_inference_steps=50
guidance_scale=7.5
prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt="bad hands, blurry"
pipe = OnnxStableDiffusionPipeline.from_pretrained("stable_diffusion_onnx", provider="DmlExecutionProvider", safety_checker=None)
image = pipe(prompt, height, width, num_inference_steps, guidance_scale, negative_prompt).images[0] 
image.save("astronaut_rides_horse.png")
