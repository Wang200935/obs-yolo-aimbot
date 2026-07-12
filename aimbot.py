#!/usr/bin/env python3
"""
OBS → YOLO Pose → AimBot → Trigger Bot
完整整合：OBS Virtual Camera + YOLOv8/11 Pose + Low-Level Mouse Simulation
包含 tkinter GUI 控制面板 + 滑鼠側鍵觸發
"""

import sys
import time
import cv2
import numpy as np
import random
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum

# ── 第三方套件 ─────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("請先安裝: pip install ultralytics opencv-python numpy")
    sys.exit(1)

# ── 平台相關：滑鼠側鍵監聽 ─────────────────────────────────────
if sys.platform == "darwin":
    try:
        import Quartz
        from Quartz import (
            CGEventTapCreate, CGEventTapEnable, CGEventGetIntegerValueField,
            kCGEventLeftMouseDown, kCGEventLeftMouseUp, kCGEventRightMouseDown,
            kCGEventRightMouseUp, kCGEventOtherMouseDown, kCGEventOtherMouseUp,
            kCGEventScrollWheel, kCGEventMouseMoved, kCGEventFlagsChanged,
            kCGEventSourceUnixProcessID, kCGEventSourceStateID, kCGEventFlagMaskControl,
            kCGEventFlagMaskAlternate, kCGEventFlagMaskShift, kCGEventFlagMaskCommand,
            kCGHeadInsertEventTap, kCGEventTapOptionDefault, kCGEventTapOptionListenOnly,
            CFMachPortCreateRunLoopSource, CFRunLoopAddSource, CFRunLoopGetCurrent,
            kCFRunLoopCommonModes, CFRunLoopRunInMode, kCFRunLoopDefaultMode,
            CGEventPost, kCGHIDEventTap, CGEventCreateMouseEvent,
            kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp,
            kCGEventRightMouseDown, kCGEventRightMouseUp, kCGEventOtherMouseDown,
            kCGEventOtherMouseUp, kCGScrollEventUnitPixel,
            kCGMouseButtonLeft, kCGMouseButtonRight, kCGMouseButtonCenter,
            kCGEventOtherMouseDown, kCGEventOtherMouseUp
        )
        HAS_QUARTZ = True
    except ImportError:
        HAS_QUARTZ = False
elif sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    HAS_WIN32 = True
else:
    try:
        from pynput import mouse as pynput_mouse
        HAS_PYNPUT = True
    except ImportError:
        HAS_PYNPUT = False

# ── 跨平台滑鼠模擬 ──────────────────────────────────────────
class MouseBackend:
    """跨平台最底層滑鼠模擬：Linux(uinput/XTest)、macOS(CoreGraphics)、Windows(SendInput)"""
    
    def __init__(self):
        self.platform = sys.platform
        self._init_backend()
    
    def _init_backend(self):
        if self.platform.startswith("linux"):
            self._init_linux()
        elif self.platform == "darwin":
            self._init_macos()
        elif self.platform == "win32":
            self._init_windows()
        else:
            raise NotImplementedError(f"Platform {self.platform} not supported")
    
    # ═══════════════════════════════════════════════════════════════
    # Linux
    # ═══════════════════════════════════════════════════════════════
    def _init_linux(self):
        try:
            import evdev
            from evdev import UInput, ecodes as e
            caps = {
                e.EV_REL: [e.REL_X, e.REL_Y, e.REL_WHEEL, e.REL_HWHEEL],
                e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE, e.BTN_SIDE, e.BTN_EXTRA],
                e.EV_SYN: [e.SYN_REPORT],
            }
            self._ui = UInput(caps, name="Virtual Mouse", bustype=e.BUS_USB,
                             vendor=0x046d, product=0xc077, version=1)
            self._backend = "uinput"
            self._ecodes = e
            print("[Mouse] Using uinput (kernel-level virtual HID)")
            return
        except (ImportError, PermissionError, OSError) as ex:
            print(f"[Mouse] uinput unavailable ({ex}), falling back to XTest")
        
        try:
            from Xlib import display
            from Xlib.ext import xtest
            self._disp = display.Display()
            self._xtest = xtest
            self._X = display.X
            self._backend = "xtest"
            print("[Mouse] Using XTest (X11 extension)")
        except ImportError:
            raise RuntimeError("Linux: need python-evdev (uinput) or python-xlib (XTest)")
    
    def _linux_move_rel(self, dx: int, dy: int):
        if self._backend == "uinput":
            self._ui.write(self._ecodes.EV_REL, self._ecodes.REL_X, dx)
            self._ui.write(self._ecodes.EV_REL, self._ecodes.REL_Y, dy)
            self._ui.syn()
        else:
            self._xtest.fake_input(self._disp, self._X.MotionNotify, x=dx, y=dy)
            self._disp.sync()
    
    def _linux_move_abs(self, x: int, y: int):
        if self._backend == "uinput":
            if hasattr(self, '_xtest'):
                self._xtest.fake_input(self._disp, self._X.MotionNotify, x=x, y=y)
                self._disp.sync()
        else:
            self._xtest.fake_input(self._disp, self._X.MotionNotify, x=x, y=y)
            self._disp.sync()
    
    def _linux_click(self, button: str = "left"):
        btn_map = {"left": 1, "middle": 2, "right": 3, "side": 8, "extra": 9}
        btn = btn_map.get(button, 1)
        if self._backend == "uinput":
            key = getattr(self._ecodes, f"BTN_{button.upper()}")
            self._ui.write(self._ecodes.EV_KEY, key, 1); self._ui.syn()
            self._ui.write(self._ecodes.EV_KEY, key, 0); self._ui.syn()
        else:
            self._xtest.fake_input(self._disp, self._X.ButtonPress, btn); self._disp.sync()
            self._xtest.fake_input(self._disp, self._X.ButtonRelease, btn); self._disp.sync()
    
    def _linux_scroll(self, dx: int = 0, dy: int = 0):
        if self._backend == "uinput":
            if dy: self._ui.write(self._ecodes.EV_REL, self._ecodes.REL_WHEEL, -dy)
            if dx: self._ui.write(self._ecodes.EV_REL, self._ecodes.REL_HWHEEL, dx)
            self._ui.syn()
        else:
            for _ in range(abs(dy)):
                b = 4 if dy > 0 else 5
                self._xtest.fake_input(self._disp, self._X.ButtonPress, b); self._disp.sync()
                self._xtest.fake_input(self._disp, self._X.ButtonRelease, b); self._disp.sync()
            for _ in range(abs(dx)):
                b = 6 if dx > 0 else 7
                self._xtest.fake_input(self._disp, self._X.ButtonPress, b); self._disp.sync()
                self._xtest.fake_input(self._disp, self._X.ButtonRelease, b); self._disp.sync()
    
    # ═══════════════════════════════════════════════════════════════
    # macOS
    # ═══════════════════════════════════════════════════════════════
    def _init_macos(self):
        try:
            import Quartz
            self._Quartz = Quartz
        except ImportError:
            raise RuntimeError("macOS: need pyobjc-framework-Quartz (pip install pyobjc-framework-Quartz)")
    
    def _macos_move_rel(self, dx: int, dy: int):
        pos = self._Quartz.CGEventGetLocation(self._Quartz.CGEventCreate(None))
        self._macos_move_abs(pos.x + dx, pos.y + dy)
    
    def _macos_move_abs(self, x: float, y: float):
        evt = self._Quartz.CGEventCreateMouseEvent(
            None, self._Quartz.kCGEventMouseMoved, (x, y), self._Quartz.kCGMouseButtonLeft
        )
        self._Quartz.CGEventPost(self._Quartz.kCGHIDEventTap, evt)
    
    def _macos_click(self, button: str = "left"):
        btn_map = {
            "left": (self._Quartz.kCGEventLeftMouseDown, self._Quartz.kCGEventLeftMouseUp, self._Quartz.kCGMouseButtonLeft),
            "right": (self._Quartz.kCGEventRightMouseDown, self._Quartz.kCGEventRightMouseUp, self._Quartz.kCGMouseButtonRight),
            "middle": (self._Quartz.kCGEventOtherMouseDown, self._Quartz.kCGEventOtherMouseUp, self._Quartz.kCGMouseButtonCenter),
        }
        down, up, btn = btn_map.get(button, btn_map["left"])
        pos = self._Quartz.CGEventGetLocation(self._Quartz.CGEventCreate(None))
        self._Quartz.CGEventPost(self._Quartz.kCGHIDEventTap,
            self._Quartz.CGEventCreateMouseEvent(None, down, pos, btn))
        self._Quartz.CGEventPost(self._Quartz.kCGHIDEventTap,
            self._Quartz.CGEventCreateMouseEvent(None, up, pos, btn))
    
    def _macos_scroll(self, dx: int = 0, dy: int = 0):
        evt = self._Quartz.CGEventCreateScrollWheelEvent(
            None, self._Quartz.kCGScrollEventUnitPixel, 2, dy, dx
        )
        self._Quartz.CGEventPost(self._Quartz.kCGHIDEventTap, evt)
    
    # ═══════════════════════════════════════════════════════════════
    # Windows
    # ═══════════════════════════════════════════════════════════════
    def _init_windows(self):
        import ctypes
        from ctypes import wintypes
        self._user32 = ctypes.windll.user32
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._setup_structs()
    
    def _setup_structs(self):
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", self._wintypes.LONG), ("dy", self._wintypes.LONG),
                ("mouseData", self._wintypes.DWORD), ("dwFlags", self._wintypes.DWORD),
                ("time", self._wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(self._wintypes.ULONG)),
            ]
        class INPUT_UNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]
        class INPUT(ctypes.Structure):
            _fields_ = [("type", self._wintypes.DWORD), ("value", INPUT_UNION)]
        
        self._MOUSEINPUT = MOUSEINPUT
        self._INPUT_UNION = INPUT_UNION
        self._INPUT = INPUT
        
        self._SendInput = self._user32.SendInput
        self._SendInput.argtypes = (self._wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self._SendInput.restype = self._wintypes.UINT
        
        self._MF = {
            "MOVE": 0x0001, "ABS": 0x8000,
            "L_DOWN": 0x0002, "L_UP": 0x0004,
            "R_DOWN": 0x0008, "R_UP": 0x0010,
            "M_DOWN": 0x0020, "M_UP": 0x0040,
            "WHEEL": 0x0800, "HWHEEL": 0x1000,
        }
        self._WHEEL_DELTA = 120
    
    def _win_move_rel(self, dx: int, dy: int):
        inp = self._INPUT(type=0, value=self._INPUT_UNION(mi=self._MOUSEINPUT(dx=dx, dy=dy, dwFlags=self._MF["MOVE"])))
        self._SendInput(1, ctypes.byref(inp), ctypes.sizeof(self._INPUT))
    
    def _win_move_abs(self, x: int, y: int):
        sw = self._user32.GetSystemMetrics(0)
        sh = self._user32.GetSystemMetrics(1)
        dx = int(x * 65535 / sw)
        dy = int(y * 65535 / sh)
        inp = self._INPUT(type=0, value=self._INPUT_UNION(mi=self._MOUSEINPUT(dx=dx, dy=dy, dwFlags=self._MF["MOVE"] | self._MF["ABS"])))
        self._SendInput(1, ctypes.byref(inp), ctypes.sizeof(self._INPUT))
    
    def _win_click(self, button: str = "left"):
        btn_map = {
            "left": (self._MF["L_DOWN"], self._MF["L_UP"]),
            "right": (self._MF["R_DOWN"], self._MF["R_UP"]),
            "middle": (self._MF["M_DOWN"], self._MF["M_UP"]),
        }
        down, up = btn_map.get(button, btn_map["left"])
        inputs = (self._INPUT * 2)(
            self._INPUT(type=0, value=self._INPUT_UNION(mi=self._MOUSEINPUT(dwFlags=down))),
            self._INPUT(type=0, value=self._INPUT_UNION(mi=self._MOUSEINPUT(dwFlags=up))),
        )
        self._SendInput(2, inputs, ctypes.sizeof(self._INPUT))
    
    def _win_scroll(self, dx: int = 0, dy: int = 0):
        if dy:
            inp = self._INPUT(type=0, value=self._INPUT_UNION(mi=self._MOUSEINPUT(dwFlags=self._MF["WHEEL"], mouseData=dy * self._WHEEL_DELTA)))
            self._SendInput(1, ctypes.byref(inp), ctypes.sizeof(self._INPUT))
        if dx:
            inp = self._INPUT(type=0, value=self._INPUT_UNION(mi=self._MOUSEINPUT(dwFlags=self._MF["HWHEEL"], mouseData=dx * self._WHEEL_DELTA)))
            self._SendInput(1, ctypes.byref(inp), ctypes.sizeof(self._INPUT))
    
    # ═══════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════
    def _get_platform_prefix(self) -> str:
        if self.platform.startswith("linux"):
            return "linux"
        elif self.platform == "darwin":
            return "macos"
        elif self.platform == "win32":
            return "win"
        return self.platform
    
    def move_rel(self, dx: int, dy: int):
        prefix = self._get_platform_prefix()
        getattr(self, f"_{prefix}_move_rel")(dx, dy)
    
    def move_abs(self, x: int, y: int):
        prefix = self._get_platform_prefix()
        getattr(self, f"_{prefix}_move_abs")(x, y)
    
    def click(self, button: str = "left"):
        prefix = self._get_platform_prefix()
        getattr(self, f"_{prefix}_click")(button)
    
    def scroll(self, dx: int = 0, dy: int = 0):
        prefix = self._get_platform_prefix()
        getattr(self, f"_{prefix}_scroll")(dx, dy)
    
    def close(self):
        if hasattr(self, '_ui'):
            self._ui.close()
        if hasattr(self, '_disp'):
            self._disp.close()


# ── 側鍵監聽器 ────────────────────────────────────────────────
class SideButtonListener:
    """跨平台滑鼠側鍵(X1/X2)監聽器"""
    
    def __init__(self, on_toggle):
        self.on_toggle = on_toggle
        self.running = False
        self.thread = None
        self._init_listener()
    
    def _init_listener(self):
        if sys.platform == "darwin" and HAS_QUARTZ:
            self._init_macos()
        elif sys.platform == "win32" and HAS_WIN32:
            self._init_windows()
        else:
            try:
                from pynput import mouse
                self._listener = mouse.Listener(on_click=self._on_click)
                self._using_pynput = True
            except ImportError:
                self._listener = None
                self._using_pynput = False
                print("[SideButton] pynput not available, side button listening disabled")
    
    # macOS: 使用 CGEventTap 監聽 OtherMouseDown (X1/X2)
    def _init_macos(self):
        self._tap = None
        self._runloop_source = None
        self._running = False
    
    def _macos_callback(self, proxy, event_type, event, refcon):
        # X1 = button 3, X2 = button 4 (OtherMouseDown/Up)
        if event_type in (self._Quartz.kCGEventOtherMouseDown, self._Quartz.kCGEventOtherMouseUp):
            button = self._Quartz.CGEventGetIntegerValueField(event, self._Quartz.kCGMouseEventButtonNumber)
            # button 3 = X1 (back), button 4 = X2 (forward)
            if event_type == self._Quartz.kCGEventOtherMouseDown and button in (3, 4):
                self.on_toggle()
        return event
    
    def _start_macos(self):
        from Quartz import (
            CGEventTapCreate, CGEventTapEnable, CFMachPortCreateRunLoopSource,
            CFRunLoopAddSource, CFRunLoopGetCurrent, kCFRunLoopCommonModes,
            kCGEventOtherMouseDown, kCGEventOtherMouseUp,
            kCGHeadInsertEventTap, kCGEventTapOptionDefault
        )
        self._tap = CGEventTapCreate(
            0, kCGHeadInsertEventTap, 0,  # session, insert point, options
            (1 << kCGEventOtherMouseDown) | (1 << kCGEventOtherMouseUp),
            self._macos_callback, None
        )
        if self._tap:
            self._runloop_source = CFMachPortCreateRunLoopSource(None, self._tap, 0)
            CFRunLoopAddSource(CFRunLoopGetCurrent(), self._runloop_source, kCFRunLoopCommonModes)
            CGEventTapEnable(self._tap, True)
            self._running = True
    
    def _stop_macos(self):
        if self._tap:
            from Quartz import CGEventTapEnable, CFRunLoopRemoveSource, CFRunLoopGetCurrent, kCFRunLoopCommonModes
            CGEventTapEnable(self._tap, False)
            CFRunLoopRemoveSource(CFRunLoopGetCurrent(), self._runloop_source, kCFRunLoopCommonModes)
            self._tap = None
            self._running = False
    
    # Windows: 使用低階鉤子
    def _init_windows(self):
        import ctypes
        from ctypes import wintypes
        
        self._user32 = ctypes.windll.user32
        self._hook = None
        self._hook_proc = None
        
        # 定義結構
        class MSLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [
                ("pt", wintypes.POINT),
                ("mouseData", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_void_p),
            ]
        
        self._MSLLHOOKSTRUCT = MSLLHOOKSTRUCT
        self._HOOKPROC = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int, wintypes.WPARAM, ctypes.POINTER(MSLLHOOKSTRUCT))
        
        self._WH_MOUSE_LL = 14
        self._WM_XBUTTONDOWN = 0x020B
        self._WM_XBUTTONUP = 0x020C
        self._XBUTTON1 = 0x0001
        self._XBUTTON2 = 0x0002
        
        def hook_proc(nCode, wParam, lParam):
            if nCode >= 0:
                if wParam == self._WM_XBUTTONDOWN:
                    info = ctypes.cast(lParam, ctypes.POINTER(self._MSLLHOOKSTRUCT)).contents
                    # mouseData 高位包含 XBUTTON1/XBUTTON2
                    if info.mouseData & 0xFFFF == self._XBUTTON1 or info.mouseData & 0xFFFF == self._XBUTTON2:
                        # 在主線程執行回調
                        self.on_toggle()
                        return 1  # 阻塞事件
            return self._user32.CallNextHookEx(self._hook, nCode, wParam, lParam)
        
        self._hook_proc = self._HOOKPROC(hook_proc)
    
    def _start_windows(self):
        self._hook = self._user32.SetWindowsHookExW(
            self._WH_MOUSE_LL, self._hook_proc, None, 0
        )
    
    def _stop_windows(self):
        if self._hook:
            self._user32.UnhookWindowsHookEx(self._hook)
            self._hook = None
    
    # pynput 通用實現
    def _on_click(self, x, y, button, pressed):
        if pressed and str(button) in ("Button.x1", "Button.x2"):
            self.on_toggle()
    
    def start(self):
        if self.running:
            return
        self.running = True
        
        if sys.platform == "darwin" and HAS_QUARTZ:
            self._start_macos()
            self._run_thread = threading.Thread(target=self._run_macos_loop, daemon=True)
            self._run_thread.start()
        elif sys.platform == "win32" and HAS_WIN32:
            self._start_windows()
            self._run_thread = threading.Thread(target=self._run_windows_loop, daemon=True)
            self._run_thread.start()
        elif hasattr(self, '_listener') and self._listener:
            self._listener.start()
    
    def _run_macos_loop(self):
        from Quartz import CFRunLoopRun
        CFRunLoopRun()
    
    def _run_windows_loop(self):
        import ctypes
        from ctypes import wintypes
        msg = wintypes.MSG()
        while self.running:
            if ctypes.windll.user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
                ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))
            else:
                break
    
    def stop(self):
        self.running = False
        if sys.platform == "darwin":
            self._stop_macos()
        elif sys.platform == "win32":
            self._stop_windows()
        elif hasattr(self, '_listener') and self._listener:
            self._listener.stop()


# ── OBS 畫面擷取 ─────────────────────────────────────────────
class OBSCapture:
    """統一介面：Virtual Camera → NDI → RTMP"""
    
    def __init__(self, prefer: str = "virtual_cam"):
        self.prefer = prefer
        self.cap = None
        self.source_type = None
    
    def _try_virtual_camera(self):
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    if w >= 640 and h >= 360:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.cap = cap
                        self.source_type = f"Virtual Camera (index {i})"
                        print(f"✅ Using {self.source_type}: {w}x{h}")
                        return True
                cap.release()
        return False
    
    def _try_ndi(self):
        try:
            import NDIlib as ndi
            if not ndi.initialize(): return False
            finder = ndi.find_create_v2()
            ndi.find_wait_for_sources(finder, 2000)
            sources = ndi.find_get_current_sources(finder)
            if sources:
                self.ndi_recv = ndi.recv_create_v3(sources[0])
                self.source_type = f"NDI: {sources[0].ndi_name}"
                print(f"✅ Using {self.source_type}")
                return True
        except Exception as e:
            print(f"NDI unavailable: {e}")
        return False
    
    def _try_rtmp(self, url="rtmp://localhost:1935/live/obs"):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
                self.source_type = f"RTMP: {url}"
                print(f"✅ Using {self.source_type}")
                return True
        return False
    
    def open(self):
        methods = {
            "virtual_cam": [self._try_virtual_camera, self._try_ndi, self._try_rtmp],
            "ndi": [self._try_ndi, self._try_virtual_camera, self._try_rtmp],
            "rtmp": [self._try_rtmp, self._try_virtual_camera, self._try_ndi],
        }
        for m in methods[self.prefer]:
            if m():
                return True
        raise RuntimeError("No OBS capture source available")
    
    def read(self):
        if self.cap:
            return self.cap.read()
        elif hasattr(self, 'ndi_recv'):
            import NDIlib as ndi
            t, v, a, m = ndi.recv_capture_v2(self.ndi_recv, 1000)
            if t == ndi.FRAME_TYPE_VIDEO:
                frame = cv2.cvtColor(np.copy(v.data), cv2.COLOR_BGRA2BGR)
                ndi.recv_free_video_v2(self.ndi_recv, v)
                return True, frame
        return False, None
    
    def close(self):
        if self.cap:
            self.cap.release()
        if hasattr(self, 'ndi_recv'):
            import NDIlib as ndi
            ndi.recv_destroy(self.ndi_recv)
            ndi.destroy()


# ── YOLO Pose 目標選擇 ────────────────────────────────────────
class TargetSelector:
    """從 YOLO Pose keypoints 選擇瞄準目標"""
    
    # COCO 17 keypoints 索引
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    def __init__(self, target_part: str = "head", confidence_threshold: float = 0.5):
        self.target_part = target_part
        self.conf_thresh = confidence_threshold
    
    def select_target(self, keypoints: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        keypoints: (N, 17, 3) - x, y, conf
        回傳: (x, y) 或 None
        """
        if keypoints is None or len(keypoints) == 0:
            return None
        
        h, w = frame_shape[:2]
        best_target = None
        best_score = 0
        
        for person_kpts in keypoints:
            if person_kpts.shape[0] != 17:
                continue
            
            x = person_kpts[:, 0]
            y = person_kpts[:, 1]
            conf = person_kpts[:, 2] if person_kpts.shape[1] == 3 else np.ones(17)
            
            target_pt = self._get_target_point(x, y, conf)
            if target_pt is None:
                continue
            
            # 評分：信心度 + 畫面中心距離
            cx, cy = w // 2, h // 2
            dist = np.hypot(target_pt[0] - cx, target_pt[1] - cy)
            avg_conf = np.mean(conf[conf > self.conf_thresh]) if np.any(conf > self.conf_thresh) else 0
            score = avg_conf * 100 - dist * 0.1
            
            if score > best_score:
                best_score = score
                best_target = target_pt
        
        return best_target
    
    def _get_target_point(self, x: np.ndarray, y: np.ndarray, conf: np.ndarray) -> Optional[Tuple[int, int]]:
        if self.target_part == "head":
            # 頭部：鼻子、雙眼、雙耳的加權平均
            head_idx = [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE, self.LEFT_EAR, self.RIGHT_EAR]
            valid = conf[head_idx] > self.conf_thresh
            if not np.any(valid):
                return None
            hx = np.average(x[head_idx][valid], weights=conf[head_idx][valid])
            hy = np.average(y[head_idx][valid], weights=conf[head_idx][valid])
            return int(hx), int(hy)
        
        elif self.target_part == "chest":
            # 胸部：雙肩中點
            if conf[self.LEFT_SHOULDER] > self.conf_thresh and conf[self.RIGHT_SHOULDER] > self.conf_thresh:
                cx = (x[self.LEFT_SHOULDER] + x[self.RIGHT_SHOULDER]) / 2
                cy = (y[self.LEFT_SHOULDER] + y[self.RIGHT_SHOULDER]) / 2
                return int(cx), int(cy)
            return None
        
        elif self.target_part == "auto":
            # 自動優先頭部，其次胸部
            head = self._get_target_point(x, y, conf)
            if head:
                return head
            return self._get_target_point_chest(x, y, conf)
        
        return None
    
    def _get_target_point_chest(self, x, y, conf):
        if conf[self.LEFT_SHOULDER] > self.conf_thresh and conf[self.RIGHT_SHOULDER] > self.conf_thresh:
            cx = (x[self.LEFT_SHOULDER] + x[self.RIGHT_SHOULDER]) / 2
            cy = (y[self.LEFT_SHOULDER] + y[self.RIGHT_SHOULDER]) / 2
            return int(cx), int(cy)
        return None


# ── 瞄準控制器 ──────────────────────────────────────────────
class AimbotController:
    """平滑移動準心到目標，自動開槍"""
    
    def __init__(self, mouse: MouseBackend, screen_w: int, screen_h: int,
                 fov_radius: int = 300, smooth_factor: float = 0.35,
                 max_step: int = 40, click_delay: float = 0.04,
                 humanize: bool = True):
        self.mouse = mouse
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.fov_radius = fov_radius
        self.smooth_factor = smooth_factor
        self.max_step = max_step
        self.click_delay = click_delay
        self.humanize = humanize
        
        self.last_click_time = 0
        self.center = (screen_w // 2, screen_h // 2)
    
    def update(self, target_pos: Tuple[int, int]) -> bool:
        """回傳是否開槍"""
        tx, ty = target_pos
        cx, cy = self.center
        
        # 計算偏移
        dx = tx - cx
        dy = ty - cy
        dist = np.hypot(dx, dy)
        
        # 超出 FOV 不鎖定
        if dist > self.fov_radius:
            return False
        
        # 平滑移動
        move_x = int(dx * self.smooth_factor)
        move_y = int(dy * self.smooth_factor)
        
        # 限制單幀最大移動
        move_x = np.clip(move_x, -self.max_step, self.max_step)
        move_y = np.clip(move_y, -self.max_step, self.max_step)
        
        # 人性化：加入微小抖動
        if self.humanize:
            move_x += random.randint(-1, 1)
            move_y += random.randint(-1, 1)
        
        if move_x != 0 or move_y != 0:
            self.mouse.move_rel(move_x, move_y)
        
        # 判斷是否開槍：準心足夠接近目標
        if dist < 8 and (time.time() - self.last_click_time) > self.click_delay:
            self.mouse.click("left")
            self.last_click_time = time.time()
            return True
        
        return False


# ── 主程式：tkinter GUI + 側鍵觸發 ────────────────────────────
@dataclass
class Config:
    # OBS
    obs_source: str = "virtual_cam"
    # YOLO
    model_path: str = "yolo11n-pose.pt"
    conf_thresh: float = 0.5
    iou_thresh: float = 0.7
    imgsz: int = 640
    device: str = "0"
    half: bool = True
    # 瞄準
    target_part: str = "head"
    fov_radius: int = 300
    smooth_factor: float = 0.35
    max_step: int = 40
    click_delay: float = 0.04
    humanize: bool = True


class AimBotGUI:
    def __init__(self):
        self.config = Config()
        self.bot = None
        self.running = False
        self.enabled = False
        self.side_listener = None
        
        self._init_gui()
        self._init_side_button()
    
    def _init_gui(self):
        self.root = tk.Tk()
        self.root.title("OBS YOLO AimBot Controller")
        self.root.geometry("480x680")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # 狀態變數
        self.status_var = tk.StringVar(value="DISABLED")
        self.fov_var = tk.IntVar(value=self.config.fov_radius)
        self.smooth_var = tk.DoubleVar(value=self.config.smooth_factor)
        self.target_var = tk.StringVar(value=self.config.target_part)
        self.device_var = tk.StringVar(value=self.config.device)
        self.model_var = tk.StringVar(value=self.config.model_path)
        self.source_var = tk.StringVar(value=self.config.obs_source)
        self.humanize_var = tk.BooleanVar(value=self.config.humanize)
        
        self._build_ui()
        
        # 定期更新狀態標籤
        self._update_status_label()
    
    def _build_ui(self):
        # ── 標題 ────────────────────────────────────────────
        title_frame = ttk.Frame(self.root, padding=10)
        title_frame.pack(fill=tk.X)
        ttk.Label(title_frame, text="🎯 OBS YOLO AimBot Controller", font=("Helvetica", 16, "bold")).pack()
        
        # ── 狀態指示 ─────────────────────────────────────────
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                      font=("Helvetica", 14, "bold"), foreground="red")
        self.status_label.pack()
        
        # 側鍵提示
        ttk.Label(status_frame, text="🖱 Press Mouse Side Button (X1/X2) to Toggle", 
                  font=("Helvetica", 9), foreground="gray").pack(pady=2)
        
        # ── 主要控制 ─────────────────────────────────────────
        ctrl_frame = ttk.LabelFrame(self.root, text="Main Controls", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.toggle_btn = ttk.Button(btn_frame, text="▶ ENABLE", command=self._toggle_enabled, 
                                     style="Accent.TButton", width=15)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="⏹ STOP", command=self._stop_bot, width=10).pack(side=tk.LEFT, padx=5)
        
        # ── 參數調整 ─────────────────────────────────────────
        param_frame = ttk.LabelFrame(self.root, text="Aim Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # FOV Radius
        ttk.Label(param_frame, text="FOV Radius (px):").grid(row=0, column=0, sticky=tk.W, pady=2)
        fov_scale = ttk.Scale(param_frame, from_=50, to=800, variable=self.fov_var, 
                              orient=tk.HORIZONTAL, length=200,
                              command=lambda v: self._on_fov_change(float(v)))
        fov_scale.grid(row=0, column=1, padx=5)
        self.fov_label = ttk.Label(param_frame, text=f"{self.config.fov_radius}px")
        self.fov_label.grid(row=0, column=2, padx=5)
        
        # Smooth Factor
        ttk.Label(param_frame, text="Smooth Factor:").grid(row=1, column=0, sticky=tk.W, pady=2)
        smooth_scale = ttk.Scale(param_frame, from_=0.05, to=0.95, variable=self.smooth_var,
                                 orient=tk.HORIZONTAL, length=200,
                                 command=lambda v: self._on_smooth_change(float(v)))
        smooth_scale.grid(row=1, column=1, padx=5)
        self.smooth_label = ttk.Label(param_frame, text=f"{self.config.smooth_factor:.2f}")
        self.smooth_label.grid(row=1, column=2, padx=5)
        
        # Target Part
        ttk.Label(param_frame, text="Target Part:").grid(row=2, column=0, sticky=tk.W, pady=2)
        target_combo = ttk.Combobox(param_frame, textvariable=self.target_var, 
                                    values=["head", "chest", "auto"], state="readonly", width=10)
        target_combo.grid(row=2, column=1, padx=5, sticky=tk.W)
        target_combo.bind("<<ComboboxSelected>>", self._on_target_change)
        
        # Click Delay
        ttk.Label(param_frame, text="Click Delay (s):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.click_delay_var = tk.DoubleVar(value=0.04)
        click_scale = ttk.Scale(param_frame, from_=0.01, to=0.2, variable=self.click_delay_var,
                                orient=tk.HORIZONTAL, length=200,
                                command=lambda v: setattr(self, 'click_delay', float(v)))
        click_scale.grid(row=3, column=1, padx=5)
        
        # Humanize
        ttk.Checkbutton(param_frame, text="Humanize Movement", variable=self.humanize_var,
                       command=self._on_humanize_change).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # ── 進階設定 ─────────────────────────────────────────
        adv_frame = ttk.LabelFrame(self.root, text="Advanced Settings", padding=10)
        adv_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(adv_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        model_combo = ttk.Combobox(adv_frame, textvariable=self.model_var,
                                   values=["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt"],
                                   state="readonly", width=20)
        model_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(adv_frame, text="Device:").grid(row=1, column=0, sticky=tk.W, pady=2)
        device_combo = ttk.Combobox(adv_frame, textvariable=self.device_var,
                                    values=["0", "cpu", "mps"], state="readonly", width=10)
        device_combo.grid(row=1, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(adv_frame, text="OBS Source:").grid(row=2, column=0, sticky=tk.W, pady=2)
        source_combo = ttk.Combobox(adv_frame, textvariable=self.source_var,
                                    values=["virtual_cam", "ndi", "rtmp"], state="readonly", width=15)
        source_combo.grid(row=2, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(adv_frame, text="Conf Thresh:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.conf_var = tk.DoubleVar(value=self.config.conf_thresh)
        ttk.Scale(adv_frame, from_=0.1, to=0.9, variable=self.conf_var, orient=tk.HORIZONTAL, length=200).grid(row=3, column=1, padx=5)
        
        ttk.Label(adv_frame, text="IOU Thresh:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.iou_var = tk.DoubleVar(value=self.config.iou_thresh)
        ttk.Scale(adv_frame, from_=0.3, to=0.9, variable=self.iou_var, orient=tk.HORIZONTAL, length=200).grid(row=4, column=1, padx=5)
        
        ttk.Label(adv_frame, text="Image Size:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.imgsz_var = tk.IntVar(value=self.config.imgsz)
        imgsz_combo = ttk.Combobox(adv_frame, textvariable=self.imgsz_var,
                                   values=[320, 480, 640, 800, 1280], state="readonly", width=10)
        imgsz_combo.grid(row=5, column=1, padx=5, sticky=tk.W)
        
        # ── 日誌區域 ─────────────────────────────────────────
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # 樣式
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Helvetica", 11, "bold"))
    
    # ── 事件處理 ────────────────────────────────────────────
    def _toggle_enabled(self):
        self.enabled = not self.enabled
        if self.bot:
            self.bot.enabled = self.enabled
        self.status_var.set("ENABLED" if self.enabled else "DISABLED")
        self.status_label.configure(foreground="green" if self.enabled else "red")
        self.toggle_btn.configure(text="⏸ DISABLE" if self.enabled else "▶ ENABLE")
        self._log(f"AimBot {'ENABLED' if self.enabled else 'DISABLED'}")
    
    def _stop_bot(self):
        if self.bot:
            self.bot.cleanup()
            self.bot = None
        self.enabled = False
        self.status_var.set("STOPPED")
        self.status_label.configure(foreground="gray")
        self.toggle_btn.configure(text="▶ ENABLE")
        self._log("AimBot STOPPED")
    
    def _start_bot(self):
        if self.bot:
            return
        
        try:
            # 更新配置
            self.config.fov_radius = self.fov_var.get()
            self.config.smooth_factor = self.smooth_var.get()
            self.config.target_part = self.target_var.get()
            self.config.device = self.device_var.get()
            self.config.model_path = self.model_var.get()
            self.config.obs_source = self.source_var.get()
            self.config.humanize = self.humanize_var.get()
            self.config.conf_thresh = self.conf_var.get()
            self.config.iou_thresh = self.iou_var.get()
            self.config.imgsz = self.imgsz_var.get()
            self.config.click_delay = self.click_delay_var.get()
            
            self.bot = AimBot(self.config)
            self.bot.enabled = False  # 預設停用，等待側鍵或按鈕啟用
            self._log("AimBot initialized successfully")
            self._log(f"Screen: {self.bot.screen_w}x{self.bot.screen_h}")
            self._log(f"Source: {self.bot.obs.source_type}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start AimBot:\n{e}")
            self._log(f"ERROR: {e}")
    
    def _on_fov_change(self, value):
        self.config.fov_radius = int(float(value))
        self.fov_label.config(text=f"{self.config.fov_radius}px")
        if self.bot:
            self.bot.aimbot.fov_radius = self.config.fov_radius
    
    def _on_smooth_change(self, value):
        self.config.smooth_factor = float(value)
        self.smooth_label.config(text=f"{self.config.smooth_factor:.2f}")
        if self.bot:
            self.bot.aimbot.smooth_factor = self.config.smooth_factor
    
    def _on_target_change(self, event=None):
        self.config.target_part = self.target_var.get()
        if self.bot:
            self.bot.target_selector.target_part = self.config.target_part
        self._log(f"Target part: {self.config.target_part}")
    
    def _on_humanize_change(self):
        self.config.humanize = self.humanize_var.get()
        if self.bot:
            self.bot.aimbot.humanize = self.config.humanize
    
    def _on_click_delay_change(self, value):
        self.config.click_delay = float(value)
        if self.bot:
            self.bot.aimbot.click_delay = self.config.click_delay
    
    # ── 側鍵觸發 ────────────────────────────────────────────
    def _init_side_button(self):
        self.side_listener = SideButtonListener(self._on_side_button)
        self.side_listener.start()
        self._log("Side button listener started (X1/X2)")
    
    def _on_side_button(self):
        """側鍵觸發回調 - 在主線程執行切換"""
        self.root.after(0, self._toggle_enabled)
    
    # ── 日誌 ────────────────────────────────────────────────
    def _log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def _update_status_label(self):
        """定期更新狀態"""
        if self.bot and self.enabled:
            if hasattr(self.bot, 'aimbot') and self.bot.aimbot.locked_target:
                self.status_var.set("LOCKED 🎯")
            else:
                self.status_var.set("ENABLED")
        self.root.after(100, self._update_status_label)
    
    # ── 主循環 ──────────────────────────────────────────────
    def run(self):
        # 啟動 AimBot 後台線程
        self._start_bot()
        
        # 啟動 GUI 主循環
        self.root.mainloop()
        
        # 清理
        if self.side_listener:
            self.side_listener.stop()
        if self.bot:
            self.bot.cleanup()
    
    def _start_bot(self):
        """在背景線程啟動 AimBot"""
        def init_bot():
            try:
                self._start_bot()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to start AimBot:\n{e}"))
        
        threading.Thread(target=init_bot, daemon=True).start()
    
    def _on_close(self):
        self._stop_bot()
        if self.side_listener:
            self.side_listener.stop()
        self.root.destroy()


# ── AimBot 核心類（適配 GUI） ─────────────────────────────────
class AimBot:
    def __init__(self, config: Config):
        self.config = config
        self.enabled = False
        self.running = False
        
        print("[Init] Loading YOLO model...")
        self.model = YOLO(config.model_path)
        # 自動轉換 device 字串: "0" -> "cuda:0"
        device = config.device
        if device.isdigit():
            device = f"cuda:{device}"
        try:
            self.model.to(device)
        except Exception as e:
            print(f"[Init] GPU not available: {e}, falling back to CPU")
            self.model.to("cpu")
            device = "cpu"
        self._device = device
        if config.half and config.device != "cpu":
            self.model.model.half()
        
        print("[Init] Opening OBS capture...")
        self.obs = OBSCapture(prefer=config.obs_source)
        self.obs.open()
        
        ret, frame = self.obs.read()
        if not ret:
            raise RuntimeError("Cannot read from OBS source")
        self.screen_h, self.screen_w = frame.shape[:2]
        print(f"[Init] Screen resolution: {self.screen_w}x{self.screen_h}")
        
        print("[Init] Initializing mouse backend...")
        self.mouse = MouseBackend()
        
        print("[Init] Setting up aim controller...")
        self.target_selector = TargetSelector(
            target_part=config.target_part,
            confidence_threshold=config.conf_thresh
        )
        self.aimbot = AimbotController(
            mouse=self.mouse,
            screen_w=self.screen_w,
            screen_h=self.screen_h,
            fov_radius=config.fov_radius,
            smooth_factor=config.smooth_factor,
            max_step=config.max_step,
            click_delay=config.click_delay,
            humanize=config.humanize,
        )
        
        self.running = True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(
            frame,
            verbose=False,
            conf=self.config.conf_thresh,
            iou=self.config.iou_thresh,
            imgsz=self.config.imgsz,
            half=self.config.half and "cuda" in getattr(self, "_device", "cpu"),
            device=getattr(self, "_device", self.config.device),
            classes=[0],
        )
        
        result = results[0]
        annotated = frame.copy()
        
        if self.config.draw_skeleton and result.keypoints is not None:
            annotated = result.plot()
        
        target_pos = None
        if result.keypoints is not None and len(result.keypoints) > 0:
            kpts = result.keypoints.xy.cpu().numpy()
            conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            
            if conf is not None:
                keypoints = np.concatenate([kpts, conf[..., None]], axis=-1)
            else:
                keypoints = kpts
            
            target_pos = self.target_selector.select_target(keypoints, frame.shape)
        
        fired = False
        if self.enabled and target_pos:
            fired = self.aimbot.update(target_pos)
            
            if target_pos:
                tx, ty = target_pos
                cv2.drawMarker(annotated, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
                cv2.putText(annotated, "TARGET", (tx + 20, ty - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if self.config.draw_fov:
            center = (self.screen_w // 2, self.screen_h // 2)
            cv2.circle(annotated, center, self.config.fov_radius, (255, 0, 0), 2)
            cv2.putText(annotated, f"FOV: {self.config.fov_radius}px", 
                       (center[0] - 80, center[1] - self.config.fov_radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        status_color = (0, 255, 0) if self.enabled else (0, 0, 255)
        status_text = "ENABLED" if self.enabled else "DISABLED"
        cv2.putText(annotated, f"AIMBOT: {status_text}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        cv2.putText(annotated, f"Target: {self.config.target_part.upper()}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"FOV: {self.config.fov_radius}px | Smooth: {self.config.smooth_factor}",
                   (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Source: {self.obs.source_type}", (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        if fired:
            cv2.putText(annotated, "FIRE!", (self.screen_w // 2 - 50, self.screen_h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        
        return annotated
    
    def cleanup(self):
        self.running = False
        self.obs.close()
        self.mouse.close()
        cv2.destroyAllWindows()
        print("[Run] Cleaned up")


def main():
    # macOS 高 DPI 支援
    if sys.platform == "darwin":
        try:
            from AppKit import NSApplication
            NSApplication.sharedApplication()
        except ImportError:
            pass
    
    app = AimBotGUI()
    app.run()


if __name__ == "__main__":
    main()