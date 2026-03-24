from typing import List, Optional

from src.experiments import distill_registry


def main(argv: Optional[List[str]] = None) -> int:
    return distill_registry.main(argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
