"""Original worker with the authorized API-model binding at its solve boundary."""
from . import worker
from .model_binding import solve


def main():
    worker.solve = solve
    return worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
