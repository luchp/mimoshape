import filecmp
from pathlib import Path

class OpenWriteChecked:
    def __init__(self, fn: Path, mode: str = "w", chmod: int = None):
        self.fn = fn
        suffix = fn.suffix or "."
        self.fn_tmp = self.fn.with_suffix(f"{suffix}_")
        self.mode = mode
        self.chmod = chmod
        self.equal = False

    def __enter__(self):
        self.open_file = self.fn_tmp.open(self.mode)
        return self

    def __exit__(self, *args):
        self.open_file.close()
        try:
            equal = filecmp.cmp(self.fn, self.fn_tmp)
        except FileNotFoundError:
            equal = False
        if equal:
            print(f"File {self.fn} untouched")
            self.fn_tmp.unlink()
        else:
            self.fn.unlink(missing_ok=True)
            self.fn_tmp.rename(self.fn)
            print(f"File {self.fn} changed!")
            if self.chmod:
                self.fn.chmod(self.chmod)
        self.equal = equal
        return None

