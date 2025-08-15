import ray
import time
import sys

def wait_nodes(expected_nodes):
    while True:
        nodes = len([n for n in ray.nodes() if n["Alive"]])
        print(f"Current nodes: {nodes}/{expected_nodes}")
        if nodes >= expected_nodes:
            print("All nodes joined")
            return True            
        time.sleep(10)

if __name__ == "__main__":
    ray.init("auto")
    success = wait_nodes(int(sys.argv[1]))
