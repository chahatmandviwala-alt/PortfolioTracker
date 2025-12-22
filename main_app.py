import subprocess
import sys
import time
import threading
import webview
import socket
import os

# -------------------------------------------------------------------
# Icon configuration
# -------------------------------------------------------------------
ICON_PATH = os.path.join(os.path.dirname(__file__), "app_icon.ico")

# Try to import Win32 modules for setting the window icon (Windows only)
try:
    import win32con
    import win32gui
    import win32api
except ImportError:
    win32con = None
    win32gui = None
    win32api = None


STREAMLIT_PORT = 8501
STREAMLIT_URL = f"http://127.0.0.1:{STREAMLIT_PORT}"


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a TCP port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_streamlit():
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "main.py",
        "--server.port",
        str(STREAMLIT_PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    # Do NOT capture stdout/stderr in a PIPE unless you read it.
    return subprocess.Popen(
        cmd, 
        stdout=subprocess.DEVNULL,   # or just remove stdout/stderr arguments entirely
        stderr=subprocess.STDOUT
    )


def wait_for_streamlit(port: int, timeout: float = 30.0):
    """Wait until Streamlit is serving HTTP on the given port."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_open(port):
            return True
        time.sleep(0.5)
    return False


def set_window_icon(title_substring: str, icon_path: str):
    """
    Set window icon using Win32 API by scanning all top-level windows and
    matching those whose title contains `title_substring` (case-insensitive).
    This is more robust than FindWindow with an exact title.
    """
    if not sys.platform.startswith("win"):
        return
    if not (win32con and win32gui and win32api):
        return
    if not os.path.exists(icon_path):
        return

    title_substring = title_substring.lower()

    def enum_and_set(_hwnd, _):
        # Only visible windows
        if not win32gui.IsWindowVisible(_hwnd):
            return True

        window_text = win32gui.GetWindowText(_hwnd)
        if not window_text:
            return True

        if title_substring in window_text.lower():
            try:
                icon = win32gui.LoadImage(
                    None,
                    icon_path,
                    win32con.IMAGE_ICON,
                    0,
                    0,
                    win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE,
                )
                if icon:
                    # Big icon (title bar / Alt+Tab)
                    win32gui.SendMessage(
                        _hwnd, win32con.WM_SETICON, win32con.ICON_BIG, icon
                    )
                    # Small icon (taskbar / small title bar)
                    win32gui.SendMessage(
                        _hwnd, win32con.WM_SETICON, win32con.ICON_SMALL, icon
                    )
            except Exception:
                pass

        return True  # continue enumeration

    # Wait for the window to appear, then scan several times
    for _ in range(20):  # ~10 seconds
        time.sleep(0.5)
        try:
            win32gui.EnumWindows(enum_and_set, None)
        except Exception:
            pass
            break


def main():
    # 1. Start Streamlit server
    proc = start_streamlit()

    # 2. Wait until it is ready
    print("Starting Streamlit server...")
    if not wait_for_streamlit(STREAMLIT_PORT):
        print("Streamlit did not start in time, exiting.")
        proc.terminate()
        sys.exit(1)

    print("Streamlit server is up, opening window.")

    # 3. Create a pywebview window pointing to the local Streamlit app
    window_title = "Portfolio Tracker"
    window = webview.create_window(
        title=window_title,
        url=STREAMLIT_URL,
        width=1200,
        height=800,
        resizable=True,
    )

    # 3b. Apply the icon using Win32 API in a background thread
    threading.Thread(
        target=set_window_icon,
        args=(window_title, ICON_PATH),
        daemon=True,
    ).start()

    def on_closed():
        """Called when the window is closed – shuts down Streamlit."""
        print("Window closed, stopping Streamlit...")
        if proc.poll() is None:
            proc.terminate()

    # Attach close handler (pywebview events API)
    try:
        window.events.closed += on_closed
    except Exception:
        # Fallback: if events API not available, we'll just terminate after webview loop
        pass

    # 4. Start window loop
    webview.start(debug=False)

    # Fallback shutdown in case events didn't trigger
    if proc.poll() is None:
        proc.terminate()


if __name__ == "__main__":
    main()
