import numpy as np
from PIL import Image
import time
import pandas as pd
import matplotlib.pyplot as plt
import platform
import psutil

def apply_gaussian_blur(image_array):
    rows, cols = image_array.shape
    output = np.zeros((rows, cols))
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image_array[i-1:i+2, j-1:j+2]
            output[i, j] = np.sum(region * kernel)
    return output

def apply_sobel_edge_detection(image_array):
    rows, cols = image_array.shape
    output = np.zeros((rows, cols))
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image_array[i-1:i+2, j-1:j+2]
            gx = np.sum(region * Kx)
            gy = np.sum(region * Ky)
            output[i, j] = np.sqrt(gx**2 + gy**2)
            
    if output.max() > 0:
        output = (output / output.max()) * 255
    return output

def get_hardware_specs():
    return {
        "CPU": platform.processor(),
        "Cores": psutil.cpu_count(logical=False),
        "Threads": psutil.cpu_count(logical=True),
        "RAM": f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB",
        "OS": platform.system()
    }

def run_benchmarks(image_path, iterations=3):
    resolutions = {
        "144p": (256, 144),
        "240p": (426, 240),
        "360p": (640, 360),
        "480p": (854, 480)
    }
    
    specs = get_hardware_specs()
    print(f"Hardware Detected")
    for k, v in specs.items(): print(f"{k}: {v}")
    
    base_img = Image.open(image_path).convert('L')
    results = []

    for name, res in resolutions.items():
        print(f"\nBenchmarking {name} ({iterations} runs)")
        resized = base_img.resize(res)
        arr = np.array(resized)
        
        blur_times = []
        sobel_times = []

        for _ in range(iterations):
            # Measure Gaussian
            t0 = time.perf_counter()
            blurred = apply_gaussian_blur(arr)
            t1 = time.perf_counter()
            blur_times.append(t1 - t0)
            
            # Measure Sobel
            t2 = time.perf_counter()
            edges = apply_sobel_edge_detection(blurred)
            t3 = time.perf_counter()
            sobel_times.append(t3 - t2)

        avg_blur = np.mean(blur_times)
        avg_sobel = np.mean(sobel_times)
        total_avg = avg_blur + avg_sobel
        
        results.append({
            "Resolution": name,
            "Pixels": res[0] * res[1],
            "Avg Blur (s)": round(avg_blur, 4),
            "Avg Sobel (s)": round(avg_sobel, 4),
            "Total Avg (s)": round(total_avg, 4)
        })

    df = pd.DataFrame(results)
    
    # Plotting: Pixels (X) vs Time (Y)
    plt.figure(figsize=(10, 6))
    plt.plot(df['Pixels'], df['Total Avg (s)'], marker='o', linestyle='-', color='r', label='Total Sequential Time')
    plt.plot(df['Pixels'], df['Avg Blur (s)'], marker='s', linestyle='--', label='Gaussian Only')
    plt.plot(df['Pixels'], df['Avg Sobel (s)'], marker='^', linestyle='--', label='Sobel Only')
    
    plt.title('Execution Time vs. Pixel Count (Sequential)')
    plt.xlabel('Total Pixels (Workload)')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig('performance_vs_pixels.png')
    
    print("\nCAT 1 RESULTS")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    run_benchmarks("input.jpg")