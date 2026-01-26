"""
Simple upload to Hugging Face - only the model file
"""

from huggingface_hub import HfApi
import os

# Initialize API with your token
TOKEN = input("Enter your Hugging Face write token: ").strip()
api = HfApi(token=TOKEN)

REPO_ID = "karthikeya09/smart_image_recognation"

# Create repo if needed
print("📦 Creating repository...")
try:
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print("✅ Repository ready!")
except Exception as e:
    print(f"Note: {e}")

# Upload model
print("🧠 Uploading model (this may take a moment)...")
api.upload_file(
    path_or_fileobj="model/checkpoints/best_model.pth",
    path_in_repo="best_model.pth",
    repo_id=REPO_ID,
    repo_type="model"
)
print("✅ Model uploaded!")

print()
print(f"🎉 Done! Visit: https://huggingface.co/{REPO_ID}")
