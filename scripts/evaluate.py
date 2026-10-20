"""Default evaluation entry; writes a checkpoint-specific provenance report."""

try:
    from .evaluate_provenance import main
except ImportError:
    from evaluate_provenance import main


if __name__ == "__main__":
    main()
