from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageURLChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from PIL import Image
from pathlib import Path

# Path to your downloaded model files
model_path = Path.home() / 'mistral_models' / 'Pixtral'

# Initialize tokenizer and model from the local files
tokenizer = MistralTokenizer.from_file(f"{model_path}/tekken.json")
model = Transformer.from_folder(model_path)

# Load your image locally or use an image URL
image_path = "./004BC09243030057/CRL1.png"  # Replace with your OCR image file path
image = Image.open(image_path)

# For local image, save it to a temporary file or convert to an image URL recognized by the model
# Here, we assume your local path is accessible as a file path by the model loader,
# else upload image to accessible URL or use pillow to convert it to bytes.

prompt = "Extract and transcribe all text visible in the image, preserving exact formatting, layout, punctuation, and capitalization."

# Create a chat completion request with image and prompt
completion_request = ChatCompletionRequest(
    messages=[
        UserMessage(content=[ImageURLChunk(image_url=str(image_path)), TextChunk(text=prompt)])
    ]
)

encoded = tokenizer.encode_chat_completion(completion_request)
images = encoded.images
tokens = encoded.tokens

# Generate the output text tokens
out_tokens, _ = generate([tokens], model, images=[images], max_tokens=1024, temperature=0.3, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)

# Decode the tokens to text
result_text = tokenizer.decode(out_tokens[0])

print("Extracted OCR Text:\n", result_text)
