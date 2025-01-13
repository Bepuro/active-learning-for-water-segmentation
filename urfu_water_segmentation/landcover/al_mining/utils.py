# al_mining/utils.py

import os
import os.path as osp

def patch_cfg(cfg):
    """
    Функция, «патчащая» конфиг для каждого AL-шага:
      1) Создаёт директорию step_{cfg.al_cycle} внутри base_work_dir.
      2) Устанавливает cfg.work_dir на эту директорию.
      3) Указывает датасету ann_file='labeled.json' в текущей step-папке.
    """

    # Создадим поддиректорию (например, work_dir/step_0/) для данного цикла
    step_dir = osp.join(cfg.base_work_dir, f"step_{cfg.al_cycle}")
    os.makedirs(step_dir, exist_ok=True)

    # Обновим cfg.work_dir, чтобы логи/чекпоинты складывались сюда
    cfg.work_dir = step_dir

    # Подменим ann_file для train_dataloader, чтобы читать "labeled.json"
    # именно из step_0, step_1 и т.д.
    cfg.train_dataloader.dataset.ann_file = osp.join(step_dir, 'labeled.json')

    # При необходимости можно также подменить evaluator, test_dataloader и т.д.
    # Например:
    # if hasattr(cfg, 'val_dataloader'):
    #     cfg.val_dataloader.dataset.ann_file = ...
    # if hasattr(cfg, 'test_dataloader'):
    #     cfg.test_dataloader.dataset.ann_file = ...

def patch_runner(runner, cfg):
    """
    Дополнительная функция, «патчащая» Runner:
      - Например, перенаправляет логи (runner._log_dir) в cfg.work_dir.
      - Можно расширить логикой, если нужно.
    """
    runner._log_dir = cfg.work_dir
