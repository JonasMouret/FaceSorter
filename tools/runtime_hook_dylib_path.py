# tools/runtime_hook_dylib_path.py
# Au runtime, pointe le loader vers les .dylib embarqués dans Contents/MacOS/lib
import os, sys, pathlib
root = pathlib.Path(getattr(sys, "_MEIPASS", pathlib.Path(__file__).resolve().parent))
libdir = root / "lib"
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
    f"{libdir}:{os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')}"
)
