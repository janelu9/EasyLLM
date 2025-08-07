import socket
import time
import sys
import argparse

def wait_for_port(ip, port, interval=1):
    print(f"Waiting for {ip}:{port} to become available...")
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1) 
                s.connect((ip, port))
            print(f"✅ Port {port} on {ip} is now open!")
            return True
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(interval)
        except socket.gaierror:
            print(f"⚠️  DNS resolution failed for {ip}. Retrying...")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(1)
        except Exception as e:
            print(f"⚠️  Unexpected error: {str(e)}. Retrying...")
            time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wait for a TCP port to become available.')
    parser.add_argument('ip', help='Target IP address or hostname')
    parser.add_argument('port', type=int, help='Target port number')
    parser.add_argument('--sleep', type=float, default=10, 
                        help='Check interval in seconds (default: 10)')
    args = parser.parse_args()
    wait_for_port(args.ip, args.port, args.sleep)