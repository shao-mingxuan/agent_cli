"""L0 CLI - ESC 打断支持。"""

import os
import select
import sys
import threading

try:
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  # pragma: no cover - 非 POSIX 平台
    _HAS_TERMIOS = False


class CancelController:
    """监听 ESC 键，用于打断正在进行的对话生成。

    后台线程把 stdin 设为 cbreak 模式（单字符读入、不回显），
    检测到单独按下 ESC 时设置取消标志。start()/stop() 成对调用，
    stop() 会恢复终端原始属性。

    注意：方向键等以 ESC 起始的 CSI 序列（ESC [ ...）会被识别并忽略，
    避免误触发。
    """

    _SELECT_TIMEOUT = 0.05

    def __init__(self):
        self._cancel = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            self._fd = sys.stdin.fileno()
        except (AttributeError, OSError, ValueError):
            self._fd = None
        self._old_termios = None
        self._active = False

    @property
    def active(self) -> bool:
        """是否正在监听。"""
        return self._active

    def start(self) -> None:
        """启动监听线程。非 TTY 或平台不支持时静默跳过。"""
        if not _HAS_TERMIOS:
            return
        if self._fd is None:
            return
        if not sys.stdin.isatty():
            return
        if self._active:
            return
        self._cancel.clear()
        self._stop.clear()
        self._old_termios = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._active = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self) -> None:
        while not self._stop.is_set():
            r, _, _ = select.select([self._fd], [], [], self._SELECT_TIMEOUT)
            if not r:
                continue
            if self._stop.is_set():
                break
            try:
                ch = os.read(self._fd, 1)
            except OSError:
                continue
            if self._stop.is_set():
                break
            if ch == b"\x1b":
                # 可能是 CSI 序列（方向键/功能键）的开头，非阻塞探测下一个字节
                r2, _, _ = select.select([self._fd], [], [], self._SELECT_TIMEOUT)
                if r2:
                    try:
                        nxt = os.read(self._fd, 1)
                    except OSError:
                        nxt = b""
                    if nxt == b"[":
                        continue  # 方向键序列，忽略
                self._cancel.set()
                break

    def is_cancelled(self) -> bool:
        """是否已触发 ESC 打断。"""
        return self._cancel.is_set()

    def cancel(self) -> None:
        """手动触发取消。"""
        self._cancel.set()

    def stop(self) -> None:
        """停止监听并恢复终端属性。"""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._drain_stdin()
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._old_termios is not None and _HAS_TERMIOS:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass
        self._old_termios = None
        self._active = False

    def _drain_stdin(self) -> None:
        """非阻塞清空 stdin 残留按键，避免监听线程卡在 os.read 上。

        监听线程 set 后可能正阻塞在 os.read；stop() 恢复终端前把这些
        残留数据读走，确保线程能及时退出、不会抢走后续 readline 的输入。
        """
        try:
            import fcntl

            fd = self._fd
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            try:
                while True:
                    if not os.read(fd, 4096):
                        break
            except (BlockingIOError, OSError):
                pass
            finally:
                fcntl.fcntl(fd, fcntl.F_SETFL, flags)
        except Exception:
            pass
