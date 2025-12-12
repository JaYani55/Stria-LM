#!/usr/bin/env python3
"""
Stria-LM API Server Runner
Starts the FastAPI server with proper error handling for common issues.
"""

import sys
import socket
import argparse


def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_process_using_port(port: int) -> str:
    """Try to find what process is using the port (Windows only)."""
    if sys.platform != "win32":
        return ""
    
    try:
        import subprocess
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.split("\n"):
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if parts:
                    pid = parts[-1]
                    # Try to get process name
                    try:
                        proc_result = subprocess.run(
                            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                            capture_output=True,
                            text=True
                        )
                        if proc_result.stdout.strip():
                            proc_name = proc_result.stdout.strip().split(",")[0].strip('"')
                            return f"PID {pid} ({proc_name})"
                    except Exception:
                        pass
                    return f"PID {pid}"
        return ""
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Run the Stria-LM API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()
    
    host = args.host
    port = args.port
    
    # Check if port is available before starting uvicorn
    if not check_port_available(host, port):
        print("\n" + "=" * 70)
        print(f"ERROR: Port {port} is already in use or blocked.")
        print("=" * 70)
        
        process_info = find_process_using_port(port)
        if process_info:
            print(f"\nProcess using port {port}: {process_info}")
        
        print("\nPossible solutions:")
        print(f"  1. Use a different port: python run_api_server.py --port 8001")
        print(f"  2. Stop the process using port {port}:")
        
        if sys.platform == "win32":
            print(f"     - Open Task Manager and end the process")
            print(f"     - Or run: netstat -ano | findstr :{port}")
            print(f"       Then: taskkill /PID <pid> /F")
        else:
            print(f"     - Run: lsof -i :{port}")
            print(f"       Then: kill <pid>")
        
        print(f"  3. Check if another Stria-LM server is already running")
        print(f"  4. Check if another application (like Docker, IIS, or another web server)")
        print(f"     is using port {port}")
        
        if sys.platform == "win32":
            print("\nNote: On Windows, some ports may be reserved by the system.")
            print("Check reserved ports with: netsh interface ipv4 show excludedportrange protocol=tcp")
        
        print("=" * 70 + "\n")
        sys.exit(1)
    
    # Port is available, start the server
    try:
        import uvicorn
        
        print(f"\n{'=' * 50}")
        print(f"Starting Stria-LM API Server")
        print(f"{'=' * 50}")
        print(f"  Host: {host}")
        print(f"  Port: {port}")
        print(f"  URL:  http://{host}:{port}")
        print(f"  Docs: http://{host}:{port}/docs")
        print(f"{'=' * 50}\n")
        
        uvicorn.run(
            "src.main:app",
            host=host,
            port=port,
            reload=args.reload
        )
        
    except OSError as e:
        error_code = getattr(e, 'winerror', None) or e.errno
        
        if error_code == 10013 or error_code == 98:  # Windows / Linux permission denied
            print("\n" + "=" * 70)
            print(f"ERROR: Permission denied when binding to {host}:{port}")
            print("=" * 70)
            print("\nThis error (WinError 10013 / errno 98) usually means:")
            print("  - The port is being used by another application")
            print("  - The port is blocked by Windows Firewall")
            print("  - The port is in Windows' excluded port range")
            print("  - You don't have permission to use this port")
            print("\nSolutions:")
            print(f"  1. Try a different port: python run_api_server.py --port 8001")
            print("  2. Run as Administrator (if port < 1024)")
            print("  3. Check Windows Firewall settings")
            print("  4. Restart Windows NAT Driver service:")
            print("     net stop winnat && net start winnat")
            print("=" * 70 + "\n")
        else:
            print(f"\nServer error: {e}")
        
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
