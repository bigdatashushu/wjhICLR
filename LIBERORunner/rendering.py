"""Process-local OSMesa compatibility for Mesa 20's unsigned-byte color buffer."""
from __future__ import annotations

import os


def configure():
    if (os.environ.get("MUJOCO_GL") != "osmesa"
            or os.environ.get("LIBERO_OSMESA_BUFFER") != "uint8"):
        return
    import numpy as np
    from mujoco.osmesa import GLContext
    from OpenGL import GL, osmesa

    def make_current(self):
        if not self._context:
            raise RuntimeError("The OSMesa context has already been freed.")
        if self._buffer.dtype != np.uint8:
            self._buffer = np.zeros((self._height, self._width, 4), dtype=np.uint8)
        success = osmesa.OSMesaMakeCurrent(
            self._context, self._buffer, GL.GL_UNSIGNED_BYTE, self._width, self._height
        )
        if not success or not osmesa.OSMesaGetCurrentContext():
            raise RuntimeError("Cannot activate the OSMesa unsigned-byte context.")

    # Patch only this process's dependency class; original source files stay intact.
    # robosuite's OSMesa context inherits this method.
    GLContext.make_current = make_current
