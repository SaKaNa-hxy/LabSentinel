"""Default training entry: disjoint filename groups with recorded manifests."""

from multiprocessing import freeze_support

try:
    from .train_grouped import main
except ImportError:
    from train_grouped import main


if __name__ == "__main__":
    freeze_support()
    main()
