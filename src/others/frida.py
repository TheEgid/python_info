import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def save_model_locally(
    model_name: str = "ai-forever/FRIDA",
    local_model_dir: str = "./local_llm_models",
    max_retries: int = 3,
    retry_delay: float = 5.0,
    verify_integrity: bool = True,
    show_progress: bool = True,
) -> str:
    """
    Сохраняет модель локально с надежной загрузкой и повторными попытками.

    Args:
        model_name: Название модели SentenceTransformer
        local_model_dir: Директория для сохранения
        max_retries: Максимальное количество попыток
        retry_delay: Задержка между попытками в секундах
        verify_integrity: Проверять целостность модели
        show_progress: Выводить прогресс в лог

    Returns:
        Путь к сохраненной модели
    """
    local_model_path = Path(local_model_dir) / model_name.replace("/", "_")

    # Проверяем наличие модели в кеше
    if local_model_path.exists() and _is_model_valid(local_model_path):
        size_mb = _get_dir_size(local_model_path) / (1024**2)
        logger.info(f"✓ Модель найдена в кеше: {local_model_path} ({size_mb:.2f} MB)")
        return str(local_model_path)

    # Создаем директорию
    local_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Проверяем место на диске
    _check_disk_space(local_model_path.parent)

    current_retry_delay = retry_delay

    # Загрузка с повторными попытками
    for attempt in range(max_retries):
        temp_model_path = None
        try:
            logger.info(f"\n{'─' * 70}")
            logger.info(f"Попытка {attempt + 1}/{max_retries}: загрузка модели '{model_name}'")
            logger.info(f"{'─' * 70}")

            with tempfile.TemporaryDirectory(prefix="model_temp_") as temp_dir:
                temp_model_path = Path(temp_dir) / "model"
                temp_model_path.mkdir(parents=True, exist_ok=True)

                if show_progress:
                    logger.info("📥 Загрузка модели...")

                # Загружаем модель
                model = _load_model_with_progress(model_name, attempt)

                if show_progress:
                    logger.info("💾 Сохранение модели...")

                model.save(str(temp_model_path))

                if show_progress:
                    logger.info("📦 Перемещение в финальную директорию...")

                if local_model_path.exists():
                    shutil.rmtree(local_model_path, ignore_errors=True)

                shutil.move(str(temp_model_path), str(local_model_path))

            # Проверяем сохранение
            if not local_model_path.exists() or not list(local_model_path.glob("*")):
                raise IOError(f"Модель не была сохранена в: {local_model_path}")

            # Проверяем целостность
            if verify_integrity:
                _verify_model_integrity(local_model_path, model_name)

            size_mb = _get_dir_size(local_model_path) / (1024**2)
            file_count = len(list(local_model_path.rglob("*")))
            logger.info(f"\n{'✓' * 35}")
            logger.info("✓ Модель успешно загружена и сохранена!")
            logger.info(f"  Путь: {local_model_path}")
            logger.info(f"  Размер: {size_mb:.2f} MB")
            logger.info(f"  Файлов: {file_count}")
            logger.info(f"{'✓' * 35}\n")

            return str(local_model_path)

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            error_type = type(e).__name__
            logger.warning(f"⚠ {error_type} при попытке {attempt + 1}: {e}")
            _handle_retry_attempt(attempt, max_retries, current_retry_delay)
            current_retry_delay *= 2

        except requests.exceptions.RequestException as e:
            logger.warning(f"🌐 Сетевая ошибка при попытке {attempt + 1}: {e}")
            _handle_retry_attempt(attempt, max_retries, current_retry_delay)
            current_retry_delay *= 2

        except IOError as e:
            logger.error(f"💾 Ошибка ввода/вывода: {e}")
            if attempt < max_retries - 1:
                logger.info(f"⏳ Повторная попытка через {current_retry_delay:.1f} сек...")
                time.sleep(current_retry_delay)
                current_retry_delay *= 2
            else:
                logger.error(f"✗ Не удалось сохранить модель после {max_retries} попыток")
                raise

        except Exception as e:
            logger.error(f"❌ Непредвиденная ошибка: {type(e).__name__}: {str(e)[:200]}")
            _handle_retry_attempt(attempt, max_retries, current_retry_delay)
            current_retry_delay *= 2

        finally:
            if temp_model_path and Path(temp_model_path).exists():
                try:
                    shutil.rmtree(temp_model_path, ignore_errors=True)
                except Exception as cleanup_error:
                    logger.debug(f"Ошибка при удалении временных файлов: {cleanup_error}")

    raise Exception(f"✗ Не удалось загрузить модель '{model_name}' после {max_retries} попыток")


def _load_model_with_progress(model_name: str, attempt: int) -> SentenceTransformer:
    """Загружает модель с отслеживанием этапов."""
    stages = [
        "Подключение к репозиторию...",
        "Загрузка конфигурации...",
        "Загрузка весов модели...",
        "Загрузка токенизатора...",
        "Инициализация модели...",
    ]

    for i, stage in enumerate(stages, 1):
        logger.info(f"  [{i}/{len(stages)}] {stage}")

    try:
        model = SentenceTransformer(model_name, device="cpu")
        logger.info("  ✓ Модель успешно загружена в памяти")
        return model
    except Exception as e:
        logger.error(f"  ✗ Ошибка загрузки: {type(e).__name__}: {e}")
        raise


def _handle_retry_attempt(attempt: int, max_retries: int, delay: float) -> None:
    """Обработка повторной попытки."""
    if attempt < max_retries - 1:
        logger.info(f"⏳ Повторная попытка через {delay:.1f} секунд...")
        time.sleep(delay)
    else:
        logger.error(f"✗ Достигнут лимит попыток ({max_retries})")


def _verify_model_integrity(model_path: Path, model_name: str) -> None:
    """Проверяет наличие критических файлов модели."""
    logger.info("🔍 Проверка целостности модели...")

    required_files = [
        "config.json",
        "sentence_bert_config.json",
        "pytorch_model.bin",
    ]

    found_files = 0
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024**2)
            logger.info(f"  ✓ {file} ({size_mb:.2f} MB)")
            found_files += 1
        else:
            logger.debug(f"  - {file} (не найден)")

    if found_files >= 2:
        logger.info("✓ Целостность подтверждена")
    else:
        logger.warning(f"⚠ Найдено только {found_files} из {len(required_files)} файлов")


def _is_model_valid(model_path: Path) -> bool:
    """Проверяет валидность сохраненной модели."""
    try:
        required = ["config.json", "sentence_bert_config.json"]
        return all((model_path / f).exists() for f in required)
    except Exception:
        return False


def _check_disk_space(path: Path, min_space_gb: float = 2.0) -> None:
    """Проверяет доступное место на диске."""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)

        logger.info(f"💾 Место на диске: {free_gb:.2f} GB свободно из {total_gb:.2f} GB")

        if free_gb < min_space_gb:
            logger.warning(f"⚠ Недостаточно места! Требуется минимум {min_space_gb} GB")
    except Exception as e:
        logger.debug(f"Не удалось проверить место на диске: {e}")


def _get_dir_size(path: Path) -> int:
    """Вычисляет размер директории в байтах."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return total


def get_frida_embeddings(
    sentences: List[str],
    model_name: str = "ai-forever/FRIDA",
    batch_size: int = 32,
    local_model_dir: str = "./local_llm_models",
    force_redownload: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    """
    Возвращает эмбеддинги предложений с использованием модели FRIDA (1536 dimensions).

    Args:
        sentences: Список предложений
        model_name: Название модели
        batch_size: Размер батча
        local_model_dir: Директория для кеширования модели
        force_redownload: Принудительно перезагрузить модель
        device: Устройство для вычисления ("cpu" или "cuda")

    Returns:
        np.ndarray: Массив эмбеддингов shape=(len(sentences), 1536), dtype=float32

    Raises:
        ValueError: Если sentences пусто
    """
    if not sentences:
        raise ValueError("Список sentences не может быть пустым")

    logger.info(f"📊 Обработка {len(sentences)} предложений...")

    try:
        # Получаем или загружаем модель
        if local_model_dir:
            model_path = _get_or_download_model(
                model_name=model_name,
                local_model_dir=local_model_dir,
                force_redownload=force_redownload,
            )
        else:
            model_path = model_name
            logger.info(f"📥 Загружаем модель '{model_name}' без кеширования...")

        # Загружаем модель
        model = SentenceTransformer(model_path, device=device)
        logger.info(f"✓ Модель загружена (устройство: {device})")

        # Получаем эмбеддинги
        logger.info(f"🔄 Кодирование {len(sentences)} предложений (батч: {batch_size})...")
        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        if embeddings.size == 0:
            raise ValueError("Получены пустые эмбеддинги")

        logger.info(f"✓ Получены эмбеддинги: shape={embeddings.shape}, dtype={embeddings.dtype}")

        # Проверяем размерность
        if embeddings.shape[1] != 1536:
            logger.warning(f"⚠ Неожиданная размерность: {embeddings.shape[1]} (ожидается 1536)")

        return embeddings

    except Exception as e:
        logger.error(f"❌ Ошибка при получении эмбеддингов: {type(e).__name__}: {e}")
        raise


def _get_or_download_model(
    model_name: str,
    local_model_dir: str,
    force_redownload: bool = False,
) -> str:
    """
    Получает или загружает модель локально.

    Args:
        model_name: Название модели
        local_model_dir: Директория для сохранения
        force_redownload: Принудительная перезагрузка

    Returns:
        Путь к модели
    """
    local_model_path = Path(local_model_dir) / model_name.replace("/", "_")

    # Проверяем кеш
    if local_model_path.exists() and _is_model_valid(local_model_path) and not force_redownload:
        size_mb = _get_dir_size(local_model_path) / (1024**2)
        logger.info(f"✓ Модель найдена в кеше ({size_mb:.2f} MB)")
        return str(local_model_path)

    # Загружаем новую модель
    logger.info(f"📥 Загружаем модель '{model_name}' из Hub...")
    return save_model_locally(
        model_name=model_name,
        local_model_dir=local_model_dir,
        verify_integrity=True,
        show_progress=True,
    )
