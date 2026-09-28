"""
Benchmark script for BeyondOrigami inference time.
Runs multiple inference passes through ComfyUI and reports timing stats.

Requires ComfyUI to be running on 127.0.0.1:8188.
"""

import base64
import time
import statistics
import comfyuiservice

DEMO_IMAGE_PATH = "templates/demo_imgs/butterfly.png"
PROMPT = "origami butterfly in a zen garden"
NUM_RUNS = 5  # Number of benchmark iterations


def load_image_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_benchmark():
    print(f"🔧 BeyondOrigami Inference Benchmark")
    print(f"   Image: {DEMO_IMAGE_PATH}")
    print(f"   Prompt: \"{PROMPT}\"")
    print(f"   Runs: {NUM_RUNS}")
    print(f"   Resolution: 512×512")
    print(f"   Sampler: LCM, 2 steps, denoise 0.4")
    print("-" * 50)

    image_b64 = load_image_as_base64(DEMO_IMAGE_PATH)
    times = []

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

        times.append(elapsed)
        print(f"✅ {elapsed:.3f}s")

    if not times:
        print("\n❌ All runs failed. Is ComfyUI running on 127.0.0.1:8188?")
        return

    print("-" * 50)
    print(f"📊 Results ({len(times)} successful runs):")
    print(f"   Average:  {statistics.mean(times):.3f}s")
    print(f"   Median:   {statistics.median(times):.3f}s")
    print(f"   Min:      {min(times):.3f}s")
    print(f"   Max:      {max(times):.3f}s")
    if len(times) > 1:
        print(f"   Std Dev:  {statistics.stdev(times):.3f}s")
    print()
    print("📋 Copy this for README:")
    print(f"   | Metric | Value |")
    print(f"   |--------|-------|")
    print(f"   | Avg. inference time | **{statistics.mean(times):.2f}s** |")
    print(f"   | Resolution | 512×512 |")
    print(f"   | Sampler | LCM (2 steps) |")
    print(f"   | Denoise | 0.4 |")


if __name__ == "__main__":
    run_benchmark()
