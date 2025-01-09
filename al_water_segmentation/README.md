# Гайд

ai_water_segmentation - измененная версия urfu_water_segmentation. Советуем так же ознакомиться с urfu_water_segmentation/README.md, тут будут только ключевые изменения.

## Окружение
Зависимости зафиксированы при помощи poetry. Для полной установки окруения достаточно написать

```sh
/usr/bin/python3.9 -m venv venv39water
```
```sh
source venv39water/bin/activate
```
```sh
pip install -U pip
```
```sh
pip install poetry==1.8.3
```

```sh
poetry install
```

## Датасет landcover.ai

Готовый датасет лежит в `/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512`, его уже можно использовать в mmsegmentation.

Для запуска обучения на этом датасете:
```sh
cd landcover
```
```sh
srun -n1 --cpus-per-task=12 --mem=45000 -p apollo  --job-name=mmsegm-water python ./train.py ./config_landcover.py
```