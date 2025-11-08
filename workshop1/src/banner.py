import os


def print_banner(file_path=None):
    if file_path is None:
        # Use resources directory for banner file
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "banner.txt")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            banner = f.read()
        print(banner)
    except FileNotFoundError:
        print("🧠  AI Assistant 🧠")  # fallback if file not found


