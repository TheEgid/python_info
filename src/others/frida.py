import logging
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# def get_frida_embeddings(
#     sentences: List[str], model_name: str = "ai-forever/FRIDA", batch_size: int = 32
# ) -> np.ndarray:
#     """
#     Возвращает эмбеддинги предложений с использованием модели FRIDA.

#     Args:
#         sentences (List[str]): Список предложений (русский или английский).
#         model_name (str): Название модели SentenceTransformer или путь к локальной модели.
#         batch_size (int): Размер батча для encode (для ускорения при больших данных).

#     Returns:
#         np.ndarray: Массив эмбеддингов shape=(len(sentences), 1536), dtype=float32
#     """
#     model: SentenceTransformer = SentenceTransformer(model_name, device="cpu")
#     embeddings: np.ndarray = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
#     return embeddings


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model_locally(
    model_name: str = "ai-forever/FRIDA",
    local_model_dir: str = "./local_models",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """
    Сохраняет модель локально для последующего использования с обработкой ошибок и повторными попытками.

    Args:
        model_name (str): Название модели SentenceTransformer.
        local_model_dir (str): Директория для сохранения модели.
        max_retries (int): Максимальное количество попыток загрузки.
        retry_delay (float): Задержка между попытками в секундах.

    Returns:
        str: Путь к сохраненной модели.

    Raises:
        Exception: Если не удалось загрузить модель после всех попыток.
    """
    local_model_path = Path(local_model_dir) / model_name.replace("/", "_")

    # Проверяем, не существует ли модель уже локально
    if local_model_path.exists():
        logger.info(f"Модель уже существует в локальной директории: {local_model_path}")
        return str(local_model_path)

    # Создаем директорию
    local_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Пытаемся загрузить и сохранить модель с повторными попытками
    for attempt in range(max_retries):
        try:
            logger.info(f"Попытка {attempt + 1}/{max_retries} загрузить модель: {model_name}")

            # Загружаем модель с таймаутом
            model = SentenceTransformer(
                model_name,
                device="cpu",
                use_auth_token=False,
            )

            logger.info(f"Сохраняем модель в: {local_model_path}")
            model.save(str(local_model_path))

            # Проверяем, что модель успешно сохранилась
            if local_model_path.exists():
                logger.info(f"Модель успешно сохранена: {local_model_path}")
                return str(local_model_path)
            else:
                raise IOError(f"Модель не была сохранена по пути: {local_model_path}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Сетевая ошибка при попытке {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Повторная попытка через {retry_delay} секунд...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Экспоненциальная backoff задержка
            else:
                logger.error(f"Не удалось загрузить модель после {max_retries} попыток")
                raise

        except IOError as e:
            logger.error(f"Ошибка ввода/вывода при сохранении модели: {e}")
            raise

        except Exception as e:
            logger.error(f"Неожиданная ошибка при попытке {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Повторная попытка через {retry_delay} секунд...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Не удалось загрузить модель после {max_retries} попыток")
                raise

    # Если дошли до этой точки, все попытки провалились
    raise Exception(f"Не удалось загрузить и сохранить модель '{model_name}' после {max_retries} попыток")


def load_local_model(local_model_path: str) -> SentenceTransformer:
    """
    Загружает локально сохраненную модель.

    Args:
        local_model_path (str): Путь к локально сохраненной модели.

    Returns:
        SentenceTransformer: Загруженная модель.
    """
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Локальная модель не найдена по пути: {local_model_path}")

    print(f"Загружаем локальную модель из: {local_model_path}")
    model = SentenceTransformer(local_model_path, device="cpu")
    return model


def get_frida_embeddings(
    sentences: List[str],
    model_name: str = "ai-forever/FRIDA",
    batch_size: int = 32,
    local_model_dir: str = None,
    force_redownload: bool = False,
) -> np.ndarray:
    """
    Возвращает эмбеддинги предложений с использованием модели FRIDA.
    Поддерживает локальное сохранение модели для повторного использования.

    Args:
        sentences (List[str]): Список предложений (русский или английский).
        model_name (str): Название модели SentenceTransformer или путь к локальной модели.
        batch_size (int): Размер батча для encode (для ускорения при больших данных).
        local_model_dir (str, optional): Директория для локального сохранения модели.
                                        Если None, модель не сохраняется локально.
        force_redownload (bool): Принудительно перезагрузить модель, даже если она уже сохранена локально.

    Returns:
        np.ndarray: Массив эмбеддингов shape=(len(sentences), 1536), dtype=float32
    """

    # Определяем путь к локальной модели
    if local_model_dir:
        local_model_path = Path(local_model_dir) / model_name.replace("/", "_")

        # Проверяем, существует ли модель локально и нужно ли ее перезагружать
        if local_model_path.exists() and not force_redownload:
            print(f"Загружаем модель из локальной директории: {local_model_path}")
            model = SentenceTransformer(str(local_model_path), device="cpu")
        else:
            print(f"Загружаем модель из Hugging Face Hub: {model_name}")
            model = SentenceTransformer(model_name, device="cpu")

            # Сохраняем модель локально
            if local_model_dir:
                print(f"Сохраняем модель в локальную директорию: {local_model_path}")
                model.save(str(local_model_path))
    else:
        # Используем модель без локального сохранения
        model = SentenceTransformer(model_name, device="cpu")

    # Получаем эмбеддинги
    embeddings = model.encode(sentences, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    return embeddings
