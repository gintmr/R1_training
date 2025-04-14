import subprocess
import time
import logging 

logging.basicConfig(filename='gpu_hunter.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def get_gpu_memory_usage():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", "1,2,3,4"],
            universal_newlines=True
        )
        return output.strip().split('\n')
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None

def check_low_usage(threshold=10):
    gpu_data = get_gpu_memory_usage()
    if not gpu_data:
        return False

    for gpu in gpu_data:
        used, total = map(int, gpu.split(', '))
        usage_percent = (used / total) * 100
        if usage_percent >= threshold:
            return False
    return True

def main():
    check_interval = 60*10  # 检查间隔（秒）
    command_to_run = "nohup bash /data/wuxinrui/EasyR1/examples/run_qwen2_5_7b_math_a100.sh &"  # 替换为需要执行的命令

    while True:
        if check_low_usage(threshold=10):
            logging.info("All GPUs have memory usage below 10%. Executing command...")
            
            subprocess.run('conda deactivate', shell=True)
            subprocess.run('conda activate easyr1-wxr', shell=True)
            subprocess.run(command_to_run, shell=True)
            # 如果只需要执行一次，可以在此处添加 break
        else:
            logging.info("GPUs are in use. Waiting...")
        
        time.sleep(check_interval)

if __name__ == "__main__":
    main()