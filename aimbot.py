#!/usr/bin/env python3
"""
OBS -> YOLO Pose -> AimBot -> Trigger Bot
完整整合：OBS Virtual Camera + YOLOv11 Pose + 低層滑鼠模擬
包含 tkinter GUI 控制面板 + 滑鼠側鍵觸發
Windows 完美運作版
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

# 第三方套件
try:
    from ultralytics import YOLO
except ImportError:
    print("請先安裝: pip install ultralytics opencv-python numpy")
    sys.exit(1)

# 平台相關：滑鼠側鍵監聽
if sys.platform == "darwin":
    try:
        import Quartz
        from Quartz import (
            CGEventTapCreate, CGEventTapEnable, CGEventGetIntegerValueField,
            kCGEventLeftMouseDown, kCGEventLeftMouseUp, kCGEventRightMouseDown,
            kCGEventRightMouseUp, kCGEventOtherMouseDown, kCGEventOtherMouseUp,
            kCGEventScrollWheel, kCGEventMouseMoved, kCGEventFlagsChanged,
            kCGHeadInsertEventTap, kCGEventTapOptionDefault, kCGEventTapOptionListenOnly,
            CFMachPortCreateRunLoopSource, CFRunLoopAddSource, CFRunLoopGetCurrent,
            kCFRunLoopCommonModes, CFRunLoopRunInMode, kCFRunLoopDefaultMode,
            CGEventPost, kCGHIDEventTap, CGEventCreateMouseEvent,
            kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp,
            kCGEventRightMouseDown, kCGEventRightMouseUp, kCGEventOtherMouseDown,
            kCGEventOtherMouseUp, kCGScrollEventUnitPixel,
            kCGMouseButtonLeft, kCGMouseButtonRight, kCGMouseButtonCenter,
            kCGEventOtherMouseDown, kCGEventOtherMouseUp,
            kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput,
            CGEventMaskBit, kCGSessionEventTap, kCGHeadInsertEventTap,
            kCGEventTapOptionDefault, CGEventTapCreate, CGEventTapEnable,
            CFMachPortCreateRunLoopSource, CFRunLoopAddSource, CFRunLoopGetCurrent,
            kCFRunLoopCommonModes, CGEventTapEnable, CGEventGetIntegerValueField,
            kCGEventOtherMouseDown, kCGEventOtherMouseUp, kCGMouseEventButtonNumber
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

# 跨平台滑鼠模擬
class MouseBackend:
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
    
    def _init_macos(self):
        try:
            import Quartz
            self._Quartz = Quartz
        except ImportError:
            raise RuntimeError("macOS: pip install pyobjc-framework-Quartz")
    
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
                ("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]
        class INPUT_UNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]
        class INPUT(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("value", INPUT_UNION)]
        
        self._MOUSEINPUT = MOUSEINPUT
        self._INPUT_UNION = INPUT_UNION
        self._INPUT = INPUT
        
        self._SendInput = ctypes.windll.user32.SendInput
        self._SendInput.argtypes = (ctypes.c_uint, ctypes.c_void_p, ctypes.c_int)
        self._SendInput.restype = ctypes.c_uint
        
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


        self._start_bot()

# 滑鼠側鍵監聽器
class SideButtonListener:
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
                from pynput import mouse as pynput_mouse
                self._listener = pynput_mouse.Listener(on_click=self._on_click)
                self._using_pynput = True
            except ImportError:
                self._listener = None
                self._using_pynput = False
                print("[SideButton] pynput not available, side button listening disabled")
    
    def _init_macos(self):
        self._tap = None
        self._runloop_source = None
        self._running = False
        self._thread = None
    
    def _macos_callback(self, proxy, event_type, event, refcon):
        if event_type in (self._Quartz.kCGEventOtherMouseDown, self._Quartz.kCGEventOtherMouseUp):
            button = self._Quartz.CGEventGetIntegerValueField(event, self._Quartz.kCGMouseEventButtonNumber)
            if button in (3, 4) and event_type == self._Quartz.kCGEventOtherMouseDown:
                import threading
                threading.Thread(target=self.on_toggle, daemon=True).start()
        return event
    
    def _macos_runloop(self):
        self._Quartz = Quartz
        self._tap = self._Quartz.CGEventTapCreate(
            self._Quartz.kCGSessionEventTap,
            self._Quartz.kCGHeadInsertEventTap,
            self._Quartz.kCGEventTapOptionDefault,
            self._Quartz.CGEventMaskBit(self._Quartz.kCGEventOtherMouseDown) | 
            self._Quartz.CGEventMaskBit(self._Quartz.kCGEventOtherMouseUp),
            self._macos_callback,
            None
        )
        if not self._tap:
            print("[SideButton] Failed to create CGEventTap (need accessibility permission)")
            return
        
        self._runloop_source = self._Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        self._Quartz.CFRunLoopAddSource(self._Quartz.CFRunLoopGetCurrent(), self._runloop_source, self._Quartz.kCFRunLoopCommonModes)
        self._Quartz.CGEventTapEnable(self._tap, True)
        self._running = True
        self._Quartz.CFRunLoopRun()
    
    def _init_windows(self):
        import ctypes
        from ctypes import wintypes
        
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32
        
        self._HOOKPROC = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        self._hook = None
        self._hook_proc = self._HOOKPROC(self._windows_hook_callback)
        
        self._WH_MOUSE_LL = 14
        self._WM_XBUTTONDOWN = 0x020B
        self._WM_XBUTTONUP = 0x020C
        self._XBUTTON1 = 0x0001
        self._XBUTTON2 = 0x0002
        self._GET_XBUTTON_WPARAM = lambda wparam: (wparam >> 16) & 0xFFFF
    
    def _windows_hook_callback(self, nCode: int, wParam: int, lParam: int) -> int:
        if nCode >= 0 and wParam == self._WM_XBUTTONDOWN:
            xbutton = self._GET_XBUTTON_WPARAM(wParam)
            if xbutton in (self._XBUTTON1, self._XBUTTON2):
                import threading
                threading.Thread(target=self.on_toggle, daemon=True).start()
        return self._user32.CallNextHookEx(self._hook, nCode, wParam, lParam)
    
    def _pynput_on_click(self, x, y, button, pressed):
        if pressed and button in (pynput_mouse.Button.x1, pynput_mouse.Button.x2):
            self.on_toggle()
    
    def start(self):
        if self.running:
            return
        
        if sys.platform == "darwin" and HAS_QUARTZ:
            self.running = True
            self._thread = threading.Thread(target=self._macos_runloop, daemon=True)
            self._thread.start()
            print("[SideButton] macOS CGEventTap started (X1/X2)")
        elif sys.platform == "win32" and HAS_WIN32:
            self._hook = self._user32.SetWindowsHookExW(
                self._WH_MOUSE_LL, self._hook_proc, 
                self._kernel32.GetModuleHandleW(None), 0
            )
            if not self._hook:
                error = ctypes.get_last_error()
                print(f"[SideButton] SetWindowsHookEx failed: {error}")
            else:
                self.running = True
                print("[SideButton] Windows Low-Level Mouse Hook installed (X1/X2)")
        elif hasattr(self, '_listener') and self._listener:
            self._listener.start()
            self.running = True
            print("[SideButton] pynput listener started (X1/X2)")
        else:
            print("[SideButton] No available backend for side button listening")
    
    def stop(self):
        if not self.running:
            return
        
        if sys.platform == "darwin" and HAS_QUARTZ and self._tap:
            self._Quartz.CGEventTapEnable(self._tap, False)
            self._Quartz.CFRunLoopStop(self._Quartz.CFRunLoopGetCurrent())
            self.running = False
        elif sys.platform == "win32" and HAS_WIN32 and self._hook:
            self._user32.UnhookWindowsHookEx(self._hook)
            self._hook = None
            self.running = False
        elif hasattr(self, '_listener') and self._listener:
            self._listener.stop()
            self.running = False
        
        print("[SideButton] Stopped")


# OBS 畫面捕獲
class OBSCapture:
    def __init__(self, prefer: str = "virtual_cam"):
        self.prefer = prefer
        self.cap = None
        self.source_type = None
        self._width = 0
        self._height = 0
    
    def open(self):
        sources = [
            ("virtual_cam", self._open_virtual_cam),
            ("ndi", self._open_ndi),
            ("rtmp", self._open_rtmp),
            ("screen", self._open_screen),
        ]
        
        ordered = []
        for name, func in sources:
            if name == self.prefer:
                ordered.insert(0, (name, func))
            else:
                ordered.append((name, func))
        
        for name, func in ordered:
            try:
                print(f"[OBS] Trying {name}...")
                if func():
                    self.source_type = name
                    print(f"[OBS] Successfully opened {name} ({self._width}x{self._height})")
                    return
            except Exception as e:
                print(f"[OBS] {name} failed: {e}")
        
        raise RuntimeError("Cannot open any OBS video source")
    
    def _open_virtual_cam(self) -> bool:
        for idx in range(5):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if self._width > 100 and self._height > 100:
                        self.cap = cap
                        return True
            cap.release()
        return False
    
    def _open_ndi(self) -> bool:
        try:
            return False
        except:
            return False
    
    def _open_rtmp(self) -> bool:
        try:
            rtmp_url = "rtmp://localhost/live/obs"
            cap = cv2.VideoCapture(rtmp_url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.cap = cap
                    return True
            cap.release()
        except:
            pass
        return False
    
    def _open_screen(self) -> bool:
        try:
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes
                
                user32 = ctypes.windll.user32
                self._width = user32.GetSystemMetrics(0)
                self._height = user32.GetSystemMetrics(1)
                
                class ScreenCapture:
                    def __init__(self, w, h):
                        self.w = w
                        self.h = h
                        self.user32 = ctypes.windll.user32
                        self.gdi32 = ctypes.windll.gdi32
                    
                    def read(self):
                        hwnd = self.user32.GetDesktopWindow()
                        hdc = self.user32.GetWindowDC(hwnd)
                        memdc = self.gdi32.CreateCompatibleDC(hdc)
                        hbitmap = self.gdi32.CreateCompatibleBitmap(hdc, self.w, self.h)
                        self.gdi32.SelectObject(memdc, hbitmap)
                        self.gdi32.BitBlt(memdc, 0, 0, self.w, self.h, hdc, 0, 0, 0x00CC0020)
                        
                        bmpinfo = ctypes.create_string_buffer(40)
                        ctypes.memmove(bmpinfo, ctypes.c_int(40), 4)
                        ctypes.memmove(bmpinfo[4:8], ctypes.c_int(self.w), 4)
                        ctypes.memmove(bmpinfo[8:12], ctypes.c_int(-self.h), 4)
                        ctypes.memmove(bmpinfo[12:14], ctypes.c_short(1), 2)
                        ctypes.memmove(bmpinfo[14:16], ctypes.c_short(32), 2)
                        
                        bits = ctypes.create_string_buffer(self.w * self.h * 4)
                        self.gdi32.GetDIBits(memdc, hbitmap, 0, self.h, bits, ctypes.byref(ctypes.c_buffer(bmpinfo)), 0)
                        
                        frame = np.frombuffer(bits, dtype=np.uint8).reshape(self.h, self.w, 4)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        
                        self.gdi32.DeleteObject(hbitmap)
                        self.gdi32.DeleteDC(memdc)
                        self.user32.ReleaseDC(hwnd, hdc)
                        
                        return True, frame
                    
                    def isOpened(self):
                        return True
                    
                    def release(self):
                        pass
                
                self.cap = ScreenCapture(self._width, self._height)
                return True
            else:
                try:
                    import mss
                    self._sct = mss.mss()
                    self._monitor = self._sct.monitors[1]
                    self._width = self._monitor["width"]
                    self._height = self._monitor["height"]
                    
                    class MSSCapture:
                        def __init__(self, sct, monitor):
                            self.sct = sct
                            self.monitor = monitor
                        
                        def read(self):
                            img = self.sct.grab(self.monitor)
                            frame = np.array(img)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                            return True, frame
                        
                        def isOpened(self):
                            return True
                        
                        def release(self):
                            pass
                    
                    self.cap = MSSCapture(self._sct, self._monitor)
                    return True
                except ImportError:
                    pass
        except Exception as e:
            print(f"[OBS] Screen capture failed: {e}")
        return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        
        if hasattr(self.cap, 'read'):
            return self.cap.read()
        
        try:
            ret, frame = self.cap.read()
            return ret, frame
        except:
            return False, None
    
    def isOpened(self) -> bool:
        return self.cap is not None and (hasattr(self.cap, 'isOpened') and self.cap.isOpened() or True)
    
    def close(self):
        if self.cap and hasattr(self.cap, 'release'):
            self.cap.release()
        if hasattr(self, '_sct'):
            self._sct.close()
        self.cap = None


# 目標選擇器
class TargetSelector:
    NOSE = 0
    LEFT_EYE = 1; RIGHT_EYE = 2
    LEFT_EAR = 3; RIGHT_EAR = 4
    LEFT_SHOULDER = 5; RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7; RIGHT_ELBOW = 8
    LEFT_WRIST = 9; RIGHT_WRIST = 10
    LEFT_HIP = 11; RIGHT_HIP = 12
    LEFT_KNEE = 13; RIGHT_KNEE = 14
    LEFT_ANKLE = 15; RIGHT_ANKLE = 16
    
    def __init__(self, target_part: str = "head", confidence_threshold: float = 0.5):
        self.target_part = target_part
        self.conf_thresh = confidence_threshold
    
    def select_target(self, keypoints: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        
        if keypoints.shape[0] < 17:
            return None
        
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        conf = keypoints[:, 2] if keypoints.shape[1] > 2 else np.ones(17)
        
        if self.target_part == "head":
            return self._get_head_target(x, y, conf)
        elif self.target_part == "chest":
            return self._get_chest_target(x, y, conf)
        elif self.target_part == "auto":
            head = self._get_head_target(x, y, conf)
            if head:
                return head
            return self._get_chest_target(x, y, conf)
        
        return None
    
    def _get_head_target(self, x, y, conf) -> Optional[Tuple[int, int]]:
        head_idx = [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE, self.LEFT_EAR, self.RIGHT_EAR]
        valid = [i for i in head_idx if conf[i] > self.conf_thresh]
        
        if not valid:
            return None
        
        weights = conf[valid]
        hx = np.average(x[valid], weights=weights)
        hy = np.average(y[valid], weights=weights)
        return int(hx), int(hy)
    
    def _get_chest_target(self, x, y, conf) -> Optional[Tuple[int, int]]:
        if conf[self.LEFT_SHOULDER] > self.conf_thresh and conf[self.RIGHT_SHOULDER] > self.conf_thresh:
            cx = (x[self.LEFT_SHOULDER] + x[self.RIGHT_SHOULDER]) / 2
            cy = (y[self.LEFT_SHOULDER] + y[self.RIGHT_SHOULDER]) / 2
            return int(cx), int(cy)
        return None


# 瞄準控制器
class AimbotController:
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
        self.locked_target = None
    
    def update(self, target_pos: Tuple[int, int]) -> bool:
        tx, ty = target_pos
        cx, cy = self.center
        
        dx = tx - cx
        dy = ty - cy
        dist = np.hypot(dx, dy)
        
        if dist > self.fov_radius:
            self.locked_target = None
            return False
        
        self.locked_target = target_pos
        
        move_x = int(dx * self.smooth_factor)
        move_y = int(dy * self.smooth_factor)
        
        move_x = np.clip(move_x, -self.max_step, self.max_step)
        move_y = np.clip(move_y, -self.max_step, self.max_step)
        
        if self.humanize:
            move_x += random.randint(-1, 1)
            move_y += random.randint(-1, 1)
        
        if move_x != 0 or move_y != 0:
            self.mouse.move_rel(move_x, move_y)
        
        if dist < 8 and (time.time() - self.last_click_time) > self.click_delay:
            self.mouse.click("left")
            self.last_click_time = time.time()
            return True
        
        return False


# 配置
@dataclass
class Config:
    obs_source: str = "virtual_cam"
    model_path: str = "yolo11n-pose.pt"
    conf_thresh: float = 0.5
    iou_thresh: float = 0.7
    imgsz: int = 640
    device: str = "0"
    half: bool = True
    target_part: str = "head"
    fov_radius: int = 300
    smooth_factor: float = 0.35
    max_step: int = 40
    click_delay: float = 0.04
    humanize: bool = True
    draw_skeleton: bool = True
    draw_fov: bool = True


# 主程式類
class AimBot:
    def __init__(self, config: Config):
        self.config = config
        self.mouse = MouseBackend()
        
        if sys.platform == "win32":
            import ctypes
            user32 = ctypes.windll.user32
            self.screen_w = user32.GetSystemMetrics(0)
            self.screen_h = user32.GetSystemMetrics(1)
        elif sys.platform == "darwin":
            import Quartz
            screen = Quartz.CGMainDisplayID()
            self.screen_w = Quartz.CGDisplayPixelsWide(screen)
            self.screen_h = Quartz.CGDisplayPixelsHigh(screen)
        else:
            try:
                from Xlib import display
                d = display.Display()
                self.screen_w = d.screen().width_in_pixels
                self.screen_h = d.screen().height_in_pixels
            except:
                self.screen_w = 1920
                self.screen_h = 1080
        
        self.obs = OBSCapture(config.obs_source)
        self.obs.open()
        
        # 初始化 YOLO (自動檢測 CPU/CUDA)
        import torch as _torch
        if _torch.cuda.is_available():
            self._device = "cuda:0"
            print(f"[Init] CUDA 可用: {_torch.cuda.get_device_name(0)}")
        else:
            self._device = "cpu"
            print("[Init] 使用 CPU (無 CUDA)")
        
        self.model = YOLO(config.model_path)
        self.model.conf = config.conf_thresh
        self.model.iou = config.iou_thresh
        self.model.max_det = 10
        
        self.target_selector = TargetSelector(config.target_part, config.conf_thresh)
        
        self.aimbot = AimbotController(
            self.mouse, self.screen_w, self.screen_h,
            fov_radius=config.fov_radius,
            smooth_factor=config.smooth_factor,
            max_step=config.max_step,
            click_delay=config.click_delay,
            humanize=config.humanize
        )
        
        self.enabled = False
        self.running = False
        self._loop_thread = None
        
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
        ]
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        results = self.model(frame, imgsz=self.config.imgsz, device=getattr(self, '_device', 'cpu'), 
                           half=False, verbose=False)
        
        annotated = frame.copy()
        fired = False
        
        if results and len(results) > 0:
            res = results[0]
            
            if self.config.draw_skeleton and res.keypoints is not None:
                kpts = res.keypoints.xy.cpu().numpy()
                kpts_conf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
                
                for i, person_kpts in enumerate(kpts):
                    if kpts_conf is not None:
                        conf = kpts_conf[i]
                    else:
                        conf = np.ones(17)
                    
                    for j, (px, py) in enumerate(person_kpts):
                        if conf[j] > self.config.conf_thresh:
                            cv2.circle(annotated, (int(px), int(py)), 4, (0, 255, 0), -1)
                    
                    for (a, b) in self.skeleton:
                        if conf[a] > self.config.conf_thresh and conf[b] > self.config.conf_thresh:
                            pt1 = (int(person_kpts[a][0]), int(person_kpts[a][1]))
                            pt2 = (int(person_kpts[b][0]), int(person_kpts[b][1]))
                            cv2.line(annotated, pt1, pt2, (255, 0, 0), 2)
            
            if res.keypoints is not None and self.enabled:
                keypoints = res.keypoints.data.cpu().numpy()
                target = self.target_selector.select_target(keypoints, frame.shape[:2])
                
                if target:
                    tx, ty = target
                    cv2.drawMarker(annotated, (tx, ty), (0, 255, 255), 
                                 cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(annotated, "TARGET", (tx + 15, ty - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    fired = self.aimbot.update(target)
        
        if self.config.draw_fov:
            center = (self.screen_w // 2, self.screen_h // 2)
            cv2.circle(annotated, center, self.config.fov_radius, (255, 0, 0), 2)
            cv2.putText(annotated, f"FOV: {self.config.fov_radius}px", 
                       (center[0] - 80, center[1] - self.config.fov_radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        status_color = (0, 255, 0) if self.enabled else (0, 0, 255)
        status_text = "瞄準: 已啟用" if self.enabled else "瞄準: 已停用"
        cv2.putText(annotated, f"AIMBOT: {status_text}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        cv2.putText(annotated, f"目標: {self.config.target_part.upper()}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"FOV: {self.config.fov_radius}px | 平滑: {self.config.smooth_factor}",
                   (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"來源: {self.obs.source_type}", (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        if fired:
            cv2.putText(annotated, "開火!", (self.screen_w // 2 - 50, self.screen_h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        
        return annotated, fired
    
    def cleanup(self):
        self.running = False
        self.obs.close()
        self.mouse.close()
        cv2.destroyAllWindows()
        print("[Run] Cleaned up")
    
    def run(self, frame_callback=None):
        """主循環：不斷讀取 OBS 畫面、推理、瞄準"""
        import threading
        print("[Run] 開始主循環...")
        self.running = True
        
        def loop():
            while self.running:
                try:
                    ret, frame = self.obs.read()
                    if not ret:
                        time.sleep(0.005)
                        continue
                    annotated, _ = self.process_frame(frame)
                    if frame_callback:
                        frame_callback(annotated)
                except Exception as e:
                    print(f"[Run] 錯誤: {e}")
                    time.sleep(0.1)
            print("[Run] 主循環結束")
        
        self._loop_thread = threading.Thread(target=loop, daemon=True)
        self._loop_thread.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, '_loop_thread') and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2.0)


# GUI
class AimBotGUI:
    def __init__(self):
        self.config = Config()
        self.bot = None
        self.running = False
        self.enabled = False
        self.side_listener = None
        
        self._preview_frame = None
        
        self._init_gui()
        self._init_side_button()
    
    def _init_gui(self):
        self.root = tk.Tk()
        self.root.title("OBS YOLO 瞄準輔助")
        self.root.geometry("500x720")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self.status_var = tk.StringVar(value="已停用")
        self.fov_var = tk.IntVar(value=self.config.fov_radius)
        self.smooth_var = tk.DoubleVar(value=self.config.smooth_factor)
        self.target_var = tk.StringVar(value=self.config.target_part)
        self.device_var = tk.StringVar(value=self.config.device)
        self.model_var = tk.StringVar(value=self.config.model_path)
        self.source_var = tk.StringVar(value=self.config.obs_source)
        self.humanize_var = tk.BooleanVar(value=self.config.humanize)
        self.draw_skeleton_var = tk.BooleanVar(value=self.config.draw_skeleton)
        self.draw_fov_var = tk.BooleanVar(value=self.config.draw_fov)
        
        self._build_ui()
        self._update_status_label()
        
        # 啟動 AimBot 主循環
        self._start_bot()
    
    def _build_ui(self):
        title_frame = ttk.Frame(self.root, padding=10)
        title_frame.pack(fill=tk.X)
        ttk.Label(title_frame, text="OBS YOLO 瞄準輔助", font=("Microsoft JhengHei", 16, "bold")).pack()
        
        status_frame = ttk.LabelFrame(self.root, text="狀態", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                      font=("Microsoft JhengHei", 14, "bold"), foreground="red")
        self.status_label.pack()
        
        ttk.Label(status_frame, text="按下滑鼠側鍵 (X1/X2) 切換啟用", 
                  font=("Microsoft JhengHei", 9), foreground="gray").pack(pady=2)
        
        ctrl_frame = ttk.LabelFrame(self.root, text="主要控制", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.toggle_btn = ttk.Button(btn_frame, text="啟用", command=self._toggle_enabled, 
                                     style="Accent.TButton", width=15)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="停止", command=self._stop_bot, width=10).pack(side=tk.LEFT, padx=5)
        
        param_frame = ttk.LabelFrame(self.root, text="瞄準參數", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(param_frame, text="偵測範圍 (px):").grid(row=0, column=0, sticky=tk.W, pady=2)
        fov_scale = ttk.Scale(param_frame, from_=50, to=800, variable=self.fov_var,
                              orient=tk.HORIZONTAL, length=220,
                              command=lambda v: self._on_fov_change(float(v)))
        fov_scale.grid(row=0, column=1, padx=5)
        self.fov_label = ttk.Label(param_frame, text=f"{self.config.fov_radius}px")
        self.fov_label.grid(row=0, column=2, padx=5)
        
        ttk.Label(param_frame, text="平滑度:").grid(row=1, column=0, sticky=tk.W, pady=2)
        smooth_scale = ttk.Scale(param_frame, from_=0.05, to=0.95, variable=self.smooth_var,
                                 orient=tk.HORIZONTAL, length=220,
                                 command=lambda v: self._on_smooth_change(float(v)))
        smooth_scale.grid(row=1, column=1, padx=5)
        self.smooth_label = ttk.Label(param_frame, text=f"{self.config.smooth_factor:.2f}")
        self.smooth_label.grid(row=1, column=2, padx=5)
        
        ttk.Label(param_frame, text="目標部位:").grid(row=2, column=0, sticky=tk.W, pady=2)
        target_combo = ttk.Combobox(param_frame, textvariable=self.target_var, 
                                    values=["head", "chest", "auto"], state="readonly", width=12)
        target_combo.grid(row=2, column=1, padx=5, sticky=tk.W)
        target_combo.bind("<<ComboboxSelected>>", self._on_target_change)
        
        ttk.Label(param_frame, text="射擊延遲 (秒):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.click_delay_var = tk.DoubleVar(value=self.config.click_delay)
        click_scale = ttk.Scale(param_frame, from_=0.01, to=0.2, variable=self.click_delay_var,
                                orient=tk.HORIZONTAL, length=220,
                                command=lambda v: self._on_click_delay_change(float(v)))
        click_scale.grid(row=3, column=1, padx=5)
        
        ttk.Checkbutton(param_frame, text="人性化移動軌跡", variable=self.humanize_var,
                       command=self._on_humanize_change).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        ttk.Checkbutton(param_frame, text="顯示骨架", variable=self.draw_skeleton_var,
                       command=self._on_draw_skeleton_change).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(param_frame, text="顯示 FOV 圓", variable=self.draw_fov_var,
                       command=self._on_draw_fov_change).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        adv_frame = ttk.LabelFrame(self.root, text="進階設定", padding=10)
        adv_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(adv_frame, text="模型:").grid(row=0, column=0, sticky=tk.W, pady=2)
        model_combo = ttk.Combobox(adv_frame, textvariable=self.model_var,
                                   values=["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt"],
                                   state="readonly", width=22)
        model_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(adv_frame, text="裝置:").grid(row=1, column=0, sticky=tk.W, pady=2)
        device_combo = ttk.Combobox(adv_frame, textvariable=self.device_var,
                                    values=["0", "cpu", "mps"], state="readonly", width=12)
        device_combo.grid(row=1, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(adv_frame, text="OBS 來源:").grid(row=2, column=0, sticky=tk.W, pady=2)
        source_combo = ttk.Combobox(adv_frame, textvariable=self.source_var,
                                    values=["virtual_cam", "ndi", "rtmp", "screen"], state="readonly", width=17)
        source_combo.grid(row=2, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(adv_frame, text="信心閾值:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.conf_var = tk.DoubleVar(value=self.config.conf_thresh)
        ttk.Scale(adv_frame, from_=0.1, to=0.9, variable=self.conf_var, orient=tk.HORIZONTAL, length=220).grid(row=3, column=1, padx=5)
        
        ttk.Label(adv_frame, text="IOU 閾值:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.iou_var = tk.DoubleVar(value=self.config.iou_thresh)
        ttk.Scale(adv_frame, from_=0.3, to=0.9, variable=self.iou_var, orient=tk.HORIZONTAL, length=220).grid(row=4, column=1, padx=5)
        
        ttk.Label(adv_frame, text="圖像尺寸:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.imgsz_var = tk.IntVar(value=self.config.imgsz)
        imgsz_combo = ttk.Combobox(adv_frame, textvariable=self.imgsz_var,
                                   values=[320, 480, 640, 800, 1280], state="readonly", width=12)
        imgsz_combo.grid(row=5, column=1, padx=5, sticky=tk.W)
        
        log_frame = ttk.LabelFrame(self.root, text="日誌", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Microsoft JhengHei", 11, "bold"))
    
    def _toggle_enabled(self):
        self.enabled = not self.enabled
        if self.bot:
            self.bot.enabled = self.enabled
        self.status_var.set("已啟用" if self.enabled else "已停用")
        self.status_label.configure(foreground="green" if self.enabled else "red")
        self.toggle_btn.configure(text="停用" if self.enabled else "啟用")
        self._log(f"瞄準輔助 {'已啟用' if self.enabled else '已停用'}")
    
    def _stop_bot(self):
        if self.bot:
            self.bot.stop()
            self.bot.cleanup()
            self.bot = None
        self.enabled = False
        self._preview_frame = None
        self.status_var.set("已停止")
        self.status_label.configure(foreground="gray")
        self.toggle_btn.configure(text="啟用")
        self._log("瞄準輔助已停止")
    
    def _show_preview(self, annotated):
        """回調：從背景線程接收預覽畫面，儲存供主線程顯示"""
        self._preview_frame = annotated.copy()
    
    def _update_preview(self):
        """主線程：定期更新 OpenCV 預覽窗口"""
        if self._preview_frame is not None:
            cv2.imshow("OBS YOLO AimBot Preview", self._preview_frame)
            cv2.waitKey(1)
        self.root.after(16, self._update_preview)  # ~60 FPS
    
    def _start_bot(self):
        if self.bot:
            return
        
        try:
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
            self.config.draw_skeleton = self.draw_skeleton_var.get()
            self.config.draw_fov = self.draw_fov_var.get()
            
            self.bot = AimBot(self.config)
            self.bot.enabled = False
            
            # 啟動主循環，預覽畫面透過 tkinter 主線程更新
            self.bot.run(frame_callback=self._show_preview)
            
            # 啟動 OpenCV 預覽窗口 (必須在主線程)
            cv2.namedWindow("OBS YOLO AimBot Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("OBS YOLO AimBot Preview", 640, 360)
            self._preview_frame = None
            self._update_preview()
            self._log("瞄準輔助初始化成功")
            self._log(f"螢幕解析度: {self.bot.screen_w}x{self.bot.screen_h}")
            self._log(f"來源: {self.bot.obs.source_type}")
        except Exception as e:
            messagebox.showerror("錯誤", f"啟動瞄準輔助失敗:\n{e}")
            self._log(f"錯誤: {e}")
    
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
        self._log(f"目標部位: {self.config.target_part}")
    
    def _on_humanize_change(self):
        self.config.humanize = self.humanize_var.get()
        if self.bot:
            self.bot.aimbot.humanize = self.config.humanize
    
    def _on_click_delay_change(self, value):
        self.config.click_delay = float(value)
        if self.bot:
            self.bot.aimbot.click_delay = self.config.click_delay
    
    def _on_draw_skeleton_change(self):
        self.config.draw_skeleton = self.draw_skeleton_var.get()
        if self.bot:
            self.bot.config.draw_skeleton = self.config.draw_skeleton
    
    def _on_draw_fov_change(self):
        self.config.draw_fov = self.draw_fov_var.get()
        if self.bot:
            self.bot.config.draw_fov = self.config.draw_fov
    
    def _init_side_button(self):
        self.side_listener = SideButtonListener(self._on_side_button)
        self.side_listener.start()
        self._log("側鍵監聽已啟動 (X1/X2)")
    
    def _on_side_button(self):
        self.root.after(0, self._toggle_enabled)
    
    def _log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def _update_status_label(self):
        if self.bot and self.enabled:
            if hasattr(self.bot, 'aimbot') and self.bot.aimbot.locked_target:
                self.status_var.set("已鎖定")
            else:
                self.status_var.set("已啟用")
        self.root.after(100, self._update_status_label)
    
    def _on_close(self):
        if self.bot:
            self.bot.stop()
            self.bot.cleanup()
        if self.side_listener:
            self.side_listener.stop()
        self._preview_frame = None
        cv2.destroyAllWindows()
        self.root.destroy()
    
    def run(self):
        # _start_bot 已在 __init__ 中調用
        self.root.mainloop()
        
        if self.side_listener:
            self.side_listener.stop()
        if self.bot:
            self.bot.stop()
            self.bot.cleanup()
        cv2.destroyAllWindows()


def main():
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