import re
import time
from pathlib import Path
from typing import List

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
    "https://en.wikipedia.org/wiki/European_Union_Agency_for_the_Space_Programme",
    "https://en.wikipedia.org/wiki/Ariane_(rocket_family)",
    "https://en.wikipedia.org/wiki/Spitzer_Space_Telescope",
    "https://en.wikipedia.org/wiki/New_Horizons",
    "https://en.wikipedia.org/wiki/Cassini–Huygens",
    "https://en.wikipedia.org/wiki/Curiosity_(rover)",
    "https://en.wikipedia.org/wiki/Perseverance_(rover)",
    "https://en.wikipedia.org/wiki/InSight",
    "https://en.wikipedia.org/wiki/OSIRIS-REx",
    "https://en.wikipedia.org/wiki/Parker_Solar_Probe",
    "https://en.wikipedia.org/wiki/BepiColombo",
    "https://en.wikipedia.org/wiki/Jupiter_Icy_Moons_Explorer",
    "https://en.wikipedia.org/wiki/Solar_Orbiter",
    "https://en.wikipedia.org/wiki/CHEOPS",
    "https://en.wikipedia.org/wiki/Gaia_(spacecraft)",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}


OUTPUT_DIR = Path("articles")
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_text(content: str) -> str:
    """Удаляет сноски [1], [2] и лишние пробелы."""
    content = re.sub(r"\[\d+\]", "", content)
    content = re.sub(r"\s+", " ", content)
    return content.strip()


def sanitize_filename(name: str) -> str:
    """Приводит название файла к безопасному виду: убирает спецсимволы."""
    name = re.sub(r"[\\/*?\"<>|:]", "_", name)
    name = name.replace("(", "").replace(")", "")
    return name


def fetch_and_clean(url: str) -> str:
    """Загружает и очищает текст статьи, без удаления всего содержимого страниц."""
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    content = soup.find("div", {"class": "mw-parser-output"})
    if not content:
        return ""

    # Удаляем только заголовки нежелательных секций, но не весь их текст
    for section_title in ["References", "Bibliography", "External links", "See also"]:
        section = content.find("span", id=section_title)
        if section:
            section.name = "deleted-section"  # заменяем тег, чтобы он не отображался при get_text

    text = content.get_text(separator=" ", strip=True)
    return clean_text(text)



def run_scraper_separate_files(urls: List[str] = URLS, output_dir: Path = OUTPUT_DIR) -> None:
    """Скачивает все статьи и сохраняет каждую в отдельный .txt файл с безопасным названием.
    Если файл уже существует или статья пуста, пропускаем.
    """
    for url in tqdm(urls, desc="Загрузка статей", ncols=80):
        try:
            raw_filename = url.rstrip("/").split("/")[-1]
            filename = sanitize_filename(raw_filename) + ".txt"
            file_path = output_dir / filename

            if file_path.exists():
                print(f"Статья уже существует, пропускаем: {filename}")
                continue

            article_text = fetch_and_clean(url)

            if not article_text.strip():
                print(f"Статья пуста, пропускаем: {filename}")
                continue

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(article_text)
            time.sleep(0.5)

        except Exception as e:
            print(f"\n Ошибка при загрузке {url}: {e}")

    print(f"\n✅ Все статьи сохранены в директории {output_dir}\n")


# def process_text_file(file_path: str, chunk_size: int = 1000) -> List[str]:
#     """
#     Читает текст из файла, очищает его и разбивает на чанки фиксированного размера.

#     Args:
#         file_path (str): Путь к текстовому файлу.
#         chunk_size (int): Размер чанка в символах.

#     Returns:
#         List[str]: Список чанков текста.
#     """
#     with open(file_path, "r", encoding="utf-8") as f:
#         text: str = f.read()
#     text = text.replace("\n", " ")
#     chunked_text: List[str] = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

#     return chunked_text
