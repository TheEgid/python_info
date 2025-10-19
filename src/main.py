import sys
from typing import NoReturn

from others.helpers import multiply
from others.wiki_scraper import run_scraper


def main() -> NoReturn:
    try:
        result = multiply(300, 400)
        run_scraper()
        print(result)
    except Exception as e:
        print("Ошибка запуска:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
