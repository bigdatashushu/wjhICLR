"""Run the existing dataset smoke checks with Terra's real runtime model binding."""
from . import smoke
from .model_binding import solve


if __name__ == "__main__":
    smoke.solve = solve
    smoke.main()
