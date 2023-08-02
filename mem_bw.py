import subprocess

def run_stream_benchmark():
    try:
        # Compile the STREAM benchmark C code
        subprocess.run("gcc -O3 -march=native -o stream stream.c", shell=True, check=True)

        # Run the STREAM benchmark and capture the output
        result = subprocess.run("./stream", shell=True, capture_output=True, text=True)
        
        # Parse the output to get the memory bandwidth
        output_lines = result.stdout.split("\n")

        # Search for the line containing "Copy Bandwidth" and extract the value
        copy_bandwidth = None
        for line in output_lines:
            if "Triad" in line:
                copy_bandwidth = float(line.split()[1])
                break

        return copy_bandwidth

    except Exception as e:
        print(f"Error running STREAM benchmark: {e}")
        return None

if __name__ == "__main__":
    bandwidth_result = run_stream_benchmark()
    if bandwidth_result is not None:
        print(f"Bandwidth: {bandwidth_result:.1f} MB/s")
