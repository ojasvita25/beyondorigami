"""
Benchmark script for BeyondOrigami inference time.
Runs multiple inference passes through ComfyUI and reports timing stats.

Requires ComfyUI to be running on 127.0.0.1:8188.
"""

import base64
import os
import time
import statistics
import comfyuiservice

DEMO_IMAGES_DIR = "templates/demo_imgs"
PROMPT = "origami style, zen garden"
NUM_RUNS = 3  # Number of benchmark iterations per image


def load_image_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_demo_images(directory: str) -> list[str]:
    """Return sorted list of image paths from the demo directory."""
    supported = ('.png', '.jpg', '.jpeg', '.webp')
    images = []
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(supported) and not f.startswith('.'):
            images.append(os.path.join(directory, f))
    return images


def run_benchmark():
    images = get_demo_images(DEMO_IMAGES_DIR)
    if not images:
        print(f"❌ No images found in {DEMO_IMAGES_DIR}/")
        return

    print(f"🔧 BeyondOrigami Inference Benchmark")
    print(f"   Images: {DEMO_IMAGES_DIR}/ ({len(images)} files)")
    print(f"   Prompt: \"{PROMPT}\"")
    print(f"   Runs per image: {NUM_RUNS}")
    print(f"   Resolution: 512×512")
    print(f"   Sampler: LCM, 2 steps, denoise 0.4")
    print("=" * 55)

    all_times = []

    for img_path in images:
        img_name = os.path.basename(img_path)
        print(f"\n📷 {img_name}")
        print("-" * 55)

        image_b64 = load_image_as_base64(img_path)
        img_times = []

        for i in range(NUM_RUNS):
            print(f"   Run {i + 1}/{NUM_RUNS}...", end=" ", flush=True)
            try:
                start = time.perf_counter()
                result = comfyuiservice.fetch_image_from_comfy(PROMPT, image_b64)
                elapsed = time.perf_counter() - start
            except ConnectionRefusedError:
                print("\n\n❌ ComfyUI is not running on 127.0.0.1:8188.")
                print("   Start ComfyUI first, then re-run this benchmark.")
                return

            if result is None:
                print("❌ Failed (no image returned)")
                continue

            img_times.append(elapsed)
            all_times.append(elapsed)
            print(f"✅ {elapsed:.3f}s")

        if img_times:
            print(f"   → {img_name} avg: {statistics.mean(img_times):.3f}s")

    if not all_times:
        print("\n❌ All runs failed. Is ComfyUI running on 127.0.0.1:8188?")
        return

    print("\n" + "=" * 55)
    print(f"📊 Overall Results ({len(all_times)} runs across {len(images)} images):")
    print(f"   Average:  {statistics.mean(all_times):.3f}s")
    print(f"   Median:   {statistics.median(all_times):.3f}s")
    print(f"   Min:      {min(all_times):.3f}s")
    print(f"   Max:      {max(all_times):.3f}s")
    if len(all_times) > 1:
        print(f"   Std Dev:  {statistics.stdev(all_times):.3f}s")
    print()
    print("📋 Copy this for README:")
    print(f"   | Metric | Value |")
    print(f"   |--------|-------|")
    print(f"   | Avg. inference time | **{statistics.mean(all_times):.2f}s** |")
    print(f"   | Median inference time | **{statistics.median(all_times):.2f}s** |")
    print(f"   | Resolution | 512×512 |")
    print(f"   | Sampler | LCM (2 steps) |")
    print(f"   | Denoise | 0.4 |")


if __name__ == "__main__":
    run_benchmark()
