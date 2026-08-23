"""Public Deployment Script for Telugu Handwritten Character Recognizer via Ngrok.

Launches a secure public HTTPS tunnel to the local Streamlit application server.
Usage:
    python run_public.py [--port 8501]
"""

import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run Telugu HCR v4 with Ngrok public tunnel")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on (default: 8501)")
    args = parser.parse_args()

    port = args.port

    try:
        from pyngrok import ngrok
    except ImportError:
        print("pyngrok is not installed. Install via: pip install pyngrok")
        sys.exit(1)

    print("Opening Ngrok HTTP tunnel...")
    try:
        public_url = ngrok.connect(port, proto="http")
        print("=" * 65)
        print("TELUGU HCR V4 WEB APP IS LIVE!")
        print(f"Public URL: {public_url}")
        print(f"Local URL:  http://localhost:{port}")
        print("=" * 65)
    except Exception as e:
        print(f"Ngrok tunnel warning: {e}")
        print("Proceeding to launch local Streamlit server directly.")

    app_path = Path(__file__).resolve().parent / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        f"--server.port={port}",
        "--server.headless=true"
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down Streamlit application server and tunnels.")

if __name__ == "__main__":
    main()
