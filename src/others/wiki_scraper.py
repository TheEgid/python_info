import re
import time

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

URLS = [
    "https://en.wikipedia.org/wiki/Space_exploration",
    "https://en.wikipedia.org/wiki/Apollo_program",
    "https://en.wikipedia.org/wiki/Hubble_Space_Telescope",
    "https://en.wikipedia.org/wiki/Mars_rover",
    "https://en.wikipedia.org/wiki/International_Space_Station",
    "https://en.wikipedia.org/wiki/SpaceX",
    "https://en.wikipedia.org/wiki/Juno_(spacecraft)",
    "https://en.wikipedia.org/wiki/Voyager_program",
    "https://en.wikipedia.org/wiki/Galileo_(spacecraft)",
    "https://en.wikipedia.org/wiki/Kepler_Space_Telescope",
    "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
    "https://en.wikipedia.org/wiki/Space_Shuttle",
    "https://en.wikipedia.org/wiki/Artemis_program",
    "https://en.wikipedia.org/wiki/Skylab",
    "https://en.wikipedia.org/wiki/NASA",
    "https://en.wikipedia.org/wiki/European_Space_Agency",
    "https://en.wikipedia.org/wiki/Ariane_(rocket_family)",
    "https://en.wikipedia.org/wiki/Spitzer_Space_Telescope",
    "https://en.wikipedia.org/wiki/New_Horizons",
    "https://en.wikipedia.org/wiki/Cassini%E2%80%93Huygens",
    "https://en.wikipedia.org/wiki/Curiosity_(rover)",
    "https://en.wikipedia.org/wiki/Perseverance_(rover)",
    "https://en.wikipedia.org/wiki/InSight",
    "https://en.wikipedia.org/wiki/OSIRIS-REx",
    "https://en.wikipedia.org/wiki/Parker_Solar_Probe",
    "https://en.wikipedia.org/wiki/BepiColombo",
    "https://en.wikipedia.org/wiki/Juice_(spacecraft)",
    "https://en.wikipedia.org/wiki/Solar_Orbiter",
    "https://en.wikipedia.org/wiki/CHEOPS_(satellite)",
    "https://en.wikipedia.org/wiki/Gaia_(spacecraft)",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}


def clean_text(content: str) -> str:
    """Удаляет сноски [1], [2] и лишние пробелы."""
    content = re.sub(r"\[\d+\]", "", content)
    content = re.sub(r"\s+", " ", content)  # сжать пробелы
    return content.strip()


def fetch_and_clean(url: str) -> str:
    """Загружает и очищает текст статьи."""
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    content = soup.find("div", {"class": "mw-parser-output"})

    for section_title in ["References", "Bibliography", "External links", "See also"]:
        section = content.find("span", id=section_title)
        if section:
            for sib in section.parent.find_next_siblings():
                sib.decompose()
            section.parent.decompose()

    text = content.get_text(separator=" ", strip=True)
    return clean_text(text)


def run_scraper(output_file: str = "llm.txt") -> None:
    """Основная функция: скачивает все статьи, очищает, сжимает и сохраняет."""
    with open(output_file, "w", encoding="utf-8") as file:
        for url in tqdm(URLS, desc="Загрузка статей", ncols=80):
            try:
                clean_article_text = fetch_and_clean(url)
                file.write(clean_article_text + "\n")
                time.sleep(0.5)
            except Exception as e:
                print(f"\n⚠️ Ошибка при загрузке {url}: {e}")

    print(f"\n✅ Все статьи записаны в {output_file}\n")

    with open(output_file, "r", encoding="utf-8") as file:
        for line in file.readlines()[:5]:
            print(line.strip())
