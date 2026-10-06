# Evaluation: сравнение реальных и синтетических EM-наборов

Скрипты сравнивают **независимые наборы двумерных изображений** EPFL/Lucchi,
Kasthuri и результатов генератора, включая разметку органелл.
Соответствие изображений между двумя наборами не требуется. Обучение
сегментатора не запускается. Каждый CLI-запуск сохраняет JSON в
`evaluation/results/report/` и выводит его в stdout. `--output` позволяет
дополнительно записать копию в прежнем формате.
Прогресс и сообщения библиотек идут в stderr.

**Оценка в признаках сегментатора и эксперимент real → synthetic:**
[TASK_EXPERIMENT.md](TASK_EXPERIMENT.md). Добавлены `em_kid`, `em_fd`, `c2st`,
`frozen_seg`, а также `shape`, `psd`, `noise`. Для legacy `UnetClass/Tiny_unet_v3`
есть адаптер и [пример конфигурации](examples/unetclass.json).
Для модели на оригинальных 10 срезах используйте
[unetclass_real_only.json](examples/unetclass_real_only.json).

### Файлы каждого запуска

```text
evaluation/results/
  report/
    <date_config_info>.json
  segmentation/
    <date_config_info>/
      command.txt
      run.json
      real/
        original/<source_filename>
        predictions/<class>/<image_key>.png
        targets/<class>/<image_key>.png
      synthetic/
        original/<source_filename>
        predictions/<class>/<image_key>.png
        targets/<class>/<image_key>.png
```

`date` — UTC-время с микросекундами, `config` — имя `--task-config` (без него
имя режима), `info` — `--run-info` (по умолчанию имя synthetic каталога).
Повторные запуски создают новые папки, предыдущие результаты не перезаписываются.
Корень можно изменить через `--results-dir /path/to/results`. По умолчанию он
находится рядом с `evaluate.py`, независимо от текущего рабочего каталога.

`frozen_seg` сохраняет **те же бинарные маски, по которым считает Dice/IoU**:
sigmoid + порог для multilabel, argmax для multiclass. `targets` — снимок GT
после применения `--mask-threshold`, а не исходные grayscale-маски. PNG: 0/255,
без resize и визуальных наложений. В JSON указаны исходные пути, пути к маскам
относительно results и их SHA256. Сохраняются классы, участвующие в расчёте;
отсутствующая разрешённая GT-маска пропускается вместе с оценкой этого класса.

В `original/` сохраняются побайтовые копии обработанных изображений: исходный
формат и разрядность (включая 16-bit TIFF), без resize и нормализации. Имена
исходных файлов и вложенные папки сохраняются; небезопасные символы имён
экранируются. `frozen_seg.<domain>.per_image[].original` содержит путь копии
относительно results и SHA256. `artifacts.saved_originals` — число копий.

`command.txt` содержит рабочий каталог и экранированную команду повторного
запуска, `run.json` — параметры и статус `running/complete/partial/failed`.
После ошибки уже полученные маски остаются для диагностики; успешный итоговый
JSON не создаётся, если вычисление не завершилось. При прерывании процесса
операционной системой статус может остаться `running`. Без `frozen_seg` папки
real/synthetic пусты; остальные метрики всё равно сохраняются в report.
Python API `evaluate(args)` остаётся вычислением в памяти; автоматическое
сохранение выполняет CLI `main`.

### Полный запуск с real-only U-Net

Из Python-окружения с установленными зависимостями:

```bash
cd /mnt/d/course_work/data/synth_rep/Synthetics
MODEL_DIR='/mnt/d/course_work/4_course/expertiments/7_sem/10_slices_exp_2/segmentation/2025_05_16_00_45_13_0_0'

python -m evaluation \
  '/mnt/d/course_work/data/SLICES/cutting data 30 slices' \
  dataset/test_dataset_no_noise_lucchi \
  --metrics frozen_seg em_kid em_fd shape psd noise \
  --task-checkpoint "$MODEL_DIR"/'model_by_config_test_1_&_tiny_unet_v3_!.pt' \
  --task-adapter evaluation.unetclass_adapter:load \
  --task-config evaluation/examples/unetclass_real_only.json \
  --model-classes mitochondria PSD vesicles axon boundaries 'mitochondrial boundaries' \
  --require-real-only \
  --run-info real30_vs_lucchi
```

Эта команда использует все изображения обоих наборов. В `cutting data 30 slices`
есть обучающие срезы данной модели: для независимой оценки замените первый путь
на отдельный real-reference без обучающих срезов и соседних пространственно
связанных областей. Список обучающих срезов есть в [TASK_EXPERIMENT.md](TASK_EXPERIMENT.md).

- Без `--max-images` используются **все** поддерживаемые изображения в `original/`
  и вложенных папках. Ошибки чтения/обязательные пропущенные маски вызывают ошибку,
  а не молчаливое уменьшение набора.
- `--batch-size` по умолчанию **16**: число изображений за один проход, а не лимит
  данных или числа batch. Последний неполный batch тоже обрабатывается.
  Для разноразмерных изображений фактические batch могут быть меньше.
- Флага `--max-batch` нет. `--batch-size 4` уменьшает расход памяти, но не число
  обработанных изображений.
- `--max-images 100` выбирает до 100 изображений **из каждого набора** по `--seed`.
  Original и segmentation сохраняются только для выбранных изображений.
- `frozen_seg` оценивает все выбранные изображения. У распределительных метрик
  есть свои подвыборки: KID использует `--kid-subset-size`/`--kid-subsets`,
  PRDC/CMMD/Vendi/joint MMD — `--max-pairwise-samples` (по умолчанию 2000).
  Эти ограничения не ограничивают `frozen_seg` или сохранение его изображений.

## 1. Установка

Рекомендуется Python 3.10–3.12 и отдельное окружение. Команды выполняются из
корня `Synthetics`:

```bash
python -m venv .venv-evaluation
# Linux / WSL:
source .venv-evaluation/bin/activate
# Windows PowerShell вместо предыдущей строки:
# .venv-evaluation\Scripts\Activate.ps1

python -m pip install -r evaluation/requirements.txt
```

Для CPU можно сначала установить CPU-сборки PyTorch:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r evaluation/requirements.txt
```

Для CUDA установите совместимую пару `torch`/`torchvision` для своей системы
по [инструкции PyTorch](https://pytorch.org/get-started/locally/), затем остальные
зависимости. `--device auto` выбирает CUDA, если она доступна, иначе CPU.
Также допустимы `--device cpu`, `--device cuda`, `--device cuda:1`.

При первом расчёте FID/KID загружаются веса Inception (~91 MiB).
При первом CMMD дополнительно загружается значительно более крупный CLIP
ViT-L/14-336 с Hugging Face; предусмотрите несколько GB места для кеша моделей.
Загрузки моделей не происходят при импорте, `--help` или `--list-metrics`.

Если нужны только метрики без нейросетей:

```bash
python -m pip install -r evaluation/requirements-lite.txt
python -m evaluation REAL SYNTH --metrics intensity lbp glcm spectrum coverage morphology
```

`requirements-lite.txt` не устанавливает PyTorch. Для режима по умолчанию
нужен полный `requirements.txt`.

## 2. Режимы

| Режим | Метрики | Маски | Веса |
|---|---|---|---|
| `basic` — по умолчанию | `fid`, `kid` | Не нужны | Inception для FID |
| `medium` | Всё из basic + `intensity`, `class_intensity`, `lbp`, `glcm`, `spectrum`, `coverage`, `morphology`, `shape`, `psd`, `noise` | Нужны | Inception для FID |
| `full` | Всё из medium + `prdc`, `cmmd`, `vendi`, `swd`, `spatial`, `joint_mmd` | Нужны | Inception и CLIP |

Basic содержит два распространённых распределительных показателя из статей.
Medium помогает определить, какие свойства генератора отличаются от реальных
данных. Full дополнительно оценивает разнообразие, охват, расположение объектов
и совместные зависимости признаков.

**Если передан `--task-checkpoint`, к выбранному режиму автоматически добавляется
`frozen_seg`.** Передайте также адаптер/config и `--model-classes`.
Явный `--metrics` заменяет весь список; `--exclude-metrics frozen_seg` позволяет
отключить inference сегментации. `--require-real-only` остаётся опциональным
(по умолчанию false); для проверки real → synthetic рекомендуется включать его.

```bash
# FID и KID — режим по умолчанию:
python -m evaluation /path/to/real /path/to/synthetic --output basic.json

python -m evaluation /path/to/real /path/to/synthetic --mode medium --output medium.json
python -m evaluation /path/to/real /path/to/synthetic --mode full --output full.json

# Полный список режимов, без чтения данных/весов:
python -m evaluation --list-metrics
python -m evaluation --help
```

Пример для существующих каталогов:

```bash
python -m evaluation \
  '/mnt/d/course_work/data/Kasthuri++/labeled_data' \
  dataset/test_dataset_no_noise_kasthuri_1031 \
  --mode medium --output kasthuri_report.json

python -m evaluation \
  '/mnt/d/course_work/data/luchii++' \
  dataset/test_dataset_no_noise_lucchi \
  --mode full --max-images 50 --seed 42 --output lucchi_report.json
```

Работает и прямой запуск из любого каталога:

```bash
python /path/to/Synthetics/evaluation/evaluate.py REAL SYNTH --mode medium
```

### Выбор отдельных метрик

`--metrics` **заменяет** состав режима. `--metrics all` эквивалентен полному
списку. `--exclude-metrics` затем исключает указанные группы:

```bash
python -m evaluation REAL SYNTH --metrics fid kid prdc
python -m evaluation REAL SYNTH --metrics cmmd
python -m evaluation REAL SYNTH --mode full --exclude-metrics cmmd
python -m evaluation REAL SYNTH --metrics spatial joint_mmd --classes mitochondria vesicles
```

SSIM, PSNR, LPIPS и DICE между произвольными real/synthetic изображениями
не вычисляются: такие пары не являются соответствующими сценами. `frozen_seg`
сравнивает предсказание модели с GT **того же изображения** внутри каждого
набора. Обучение модели в CLI не запускается.

`full` включает метрики без пользовательского checkpoint. `--metrics all`
дополнительно включает `em_kid`, `em_fd`, `c2st`, `frozen_seg`: нужны checkpoint,
адаптер, порядок каналов и source groups (для C2ST), подробнее в
[TASK_EXPERIMENT.md](TASK_EXPERIMENT.md). DINO/SwAV, FRD и S₂ не реализованы.
`shape` включает Euler number и скелетные дескрипторы, но не полный анализ 3D-топологии.

## 3. Быстрые эксперименты, кеш и offline

```bash
python -m evaluation REAL SYNTH --mode medium \
  --max-images 30 --seed 0 --batch-size 8 \
  --kid-subsets 10 --kid-subset-size 100 --output trial.json
```

- `--max-images N` выбирает до N изображений **из каждого** набора независимо.
  По умолчанию берутся все изображения. Выборка воспроизводима при одинаковых
  seed и списках файлов; пути выбранных файлов записываются в JSON.
- `--batch-size 16` — размер батча экстрактора. При нехватке памяти уменьшите
  до 4 или 1. Сырые изображения не складываются целиком в один батч: сначала
  каждое приводится к входному размеру модели.
- `--torch-threads 4` ограничивает потоки CPU экстрактора.
- `--kid-subsets 50`, `--kid-subset-size 1000` задают повторные подвыборки KID.
  Фактический размер уменьшается до размера меньшего набора.
- `--max-pairwise-samples 2000` ограничивает квадратичные PRDC, CMMD, Vendi и
  joint MMD воспроизводимой выборкой. FID и KID используют все изображения,
  выбранные `--max-images`. Индексы/ключи дополнительных выборок сохраняются.
- PRDC требует хотя бы `--prdc-k + 1` изображений с каждой стороны; default k=5.
  Меньшие наборы дают `insufficient_samples`, а не автоматически изменённый k.

Нейросетевые признаки кешируются по умолчанию в
`~/.cache/synthetics-evaluation/v2`. Путь можно изменить:

```bash
python -m evaluation REAL SYNTH --mode full --cache-dir /path/to/feature-cache
```

Один и тот же Inception-кеш используется для FID, KID, PRDC и Vendi; CLIP-кеш
— для CMMD. При сравнении нескольких вариантов генератора признаки реального
набора переиспользуются. На cache hit модель не создаётся и веса не загружаются.
Соответствующие Python-зависимости всё равно должны быть установлены.

Ключ кеша включает упорядоченный список абсолютных путей, SHA-256 **содержимого**
каждого изображения, шкалу яркости, идентификатор/ревизию модели, preprocessing,
устройство и версии библиотек. Изменение изображения сбрасывает кеш даже при
сохранённых mtime и размере файла. Проверка кеша требует прочитать файлы для
хеширования. Кеш сохраняется атомарно, без pickle; повреждённый кеш пересчитывается.
Маски и классические дескрипторы не кешируются. Старые файлы кеша автоматически
не удаляются. `--no-cache` запрещает чтение и запись кеша признаков.

Кеш **весов** отдельный: Inception использует `TORCH_HOME`, CLIP — стандартный
кеш Hugging Face (`HF_HOME`). Веса выбираются фиксированно:
`pt_inception-2015-12-05-6726825d.pth` и
`openai/clip-vit-large-patch14-336` с закреплённой git-ревизией.

```bash
# После первого успешного запуска с интернетом:
python -m evaluation REAL SYNTH --mode full --offline --output offline.json
```

При отсутствии пакетов, устройства или весов запуск по умолчанию завершается
ошибкой с кодом 2. Для явно частичного отчёта:

```bash
python -m evaluation REAL SYNTH --mode full --offline \
  --skip-unavailable --output partial.json
```

Тогда отсутствующие нейросетевые метрики получают `status: unavailable`,
`value: null` и причину; весь отчёт имеет `status: partial`. Никакой замены
Inception/CLIP случайными весами или другим экстрактором нет. Неверные данные,
маски и параметры остаются ошибками даже с `--skip-unavailable`.

## 4. Данные

```text
real_or_synthetic/
  original/
    image001.png
  mitochondria/
    image001.png
  vesicles/
    image001.png
```

Внутри каждого набора маска сопоставляется с изображением по относительному пути
без расширения. Поддерживается Lucchi: `testing0000.png` ↔ `predict_test0000.png`.
Названия классов произвольные; регистр, пробелы и подчёркивания нормализуются:
`mitochondrial boundaries` = `mitochondrial_boundaries`.

По умолчанию используются общие классы, различия перечисляются в `warnings`.
Явный `--classes mitochondria vesicles` требует наличия этих классов в обоих
наборах. Каталоги без совпадающих масок игнорируются с предупреждением.
Это позволяет пропускать служебные каталоги; проверяйте `config.classes`.
Basic не сканирует каталоги масок.

Поддерживаются PNG, одностраничные TIFF, JPEG, BMP; вложенные каталоги разрешены.
Ожидаются серые изображения или равные RGB-каналы. LA/RGBA-маски компонуются
на чёрном фоне с учётом alpha. Непрозрачный LA/RGBA для исходных изображений
разрешён; прозрачные исходные изображения и цветовые маски отклоняются.
Многосрезовые TIFF заранее экспортируйте в отдельные 2D-срезы.

- Яркость делится на `--intensity-max`, default 255. Для общей uint16-шкалы
  используйте 65535, для float [0,1] — 1. Значения вне диапазона — ошибка.
  Индивидуальной min-max нормализации нет. Разные кодировки двух наборов
  предварительно приведите к общей шкале.
- Foreground маски: `value > --mask-threshold`, default 0; подходят 0/1 и 0/255.
  При интерполированных масках 0..255 можно задать 127. Порог применяется
  в исходных единицах одинаково к обоим наборам.
- Размер маски обязан совпадать с изображением. Пустая маска валидна,
  отсутствующая по умолчанию вызывает ошибку. `--allow-missing-masks`
  исключает отсутствующие пары из метрик класса и перечисляет пропуски.
  Joint MMD использует только изображения со всеми выбранными масками.
- Положительные instance-ID объединяются в бинарную маску. Компоненты —
  8-связные области, а не гарантированно отдельные органеллы.

## 5. Интерпретация метрик

Все расстояния: **меньше — ближе**, но общего порога хорошей синтетики нет.
Средние и количества — описательные статистики. PRDC и Vendi имеют другой смысл.

| Ключ JSON | Определение и ограничения |
|---|---|
| `fid` | Fréchet distance по средним и ковариациям 2048-мерных признаков **FID Inception**, не обычного torchvision Inception. По одному признаку на целое изображение. Resize 299×299 bilinear, RGB повторением серого канала, вход [−1,1]. Нужно ≥2 изображения с каждой стороны. При малой выборке значение нестабильно; rank-deficient covariance отмечается предупреждением. |
| `kid` | Несмещённая оценка MMD² с ядром `(x·y/d + 1)^3` на тех же Inception-признаках. Выводятся **неумноженные** mean/std по подвыборкам. Отрицательные оценки допустимы. std — разброс подвыборок, не доверительный интервал. Нужно ≥2 изображения. |
| `prdc` | Generative Precision, Recall, Density, Coverage по Inception-признакам. Показывают близость к реальному распределению и его охват. Строгое сравнение `< radius`, как в опубликованной реализации; дубликаты могут дать нулевые радиусы. Density может превышать 1. Больше не означает доказанно больший DICE. |
| `cmmd` | RBF MMD² по L2-нормированным CLIP ViT-L/14-336 признакам: sigma=10, scale=1000. Используется **biased V-statistic с диагональю**, как в выпущенном коде Google CMMD. Resize целого изображения до 336×336 bicubic с antialias, без center crop, CLIP mean/std. PyTorch preprocessing не гарантирует битового совпадения с JAX-референсом; вариант записан в JSON. Это не CLIPScore и текст не нужен. |
| `vendi` | Разнообразие каждого набора по cosine-kernel Inception-признаков, отдельно real/synthetic. Не мера реалистичности: шум может быть разнообразен. Сравнивайте при сопоставимом числе изображений. |
| `intensity`, `class_intensity` | Гистограммы яркости по 256 bins в [0,1], глобально/внутри класса. JS distance (корень divergence, base=2), Hellinger, total variation, Wasserstein по центрам bins и максимум разности CDF. Это не p-value KS-теста. Большие изображения/маски имеют больший вес. |
| `lbp` | Uniform LBP, 8 соседей, радиус 1, 10 bins; однопиксельная рамка исключена. JS/Hellinger/TV между объединёнными гистограммами. |
| `glcm` | 32 уровня; расстояния 1/2/4 px; направления 0/45/90/135°. Wasserstein между распределениями contrast/dissimilarity/homogeneity/ASM/energy/correlation по изображениям, отдельно для направлений. |
| `spectrum` | Нормированный радиальный спектр мощности, 64 bins в (0,0.5] cycles/pixel: вычитание среднего, окно Hann, исключение DC и угловых частот. Сравнивает форму спектра, не полный контраст; постоянные изображения исключаются с указанием количества. |
| `coverage` | Wasserstein между долями пикселей класса по изображениям. **Не** generative coverage из PRDC. |
| `morphology` | Wasserstein по плотности компонент на Mpx, площади px², эквивалентному диаметру px, eccentricity, solidity. Компоненты у края включены в плотность, но исключены из формы как обрезанные; их число выводится. |
| `swd` | **Наш вариант SWD**: случайные сырые серые патчи 7×7 из Gaussian-пирамиды масштабов 1/2/4, 128 случайных единичных проекций, усреднённый Wasserstein-1. Ограниченный равномерный reservoir кандидатов; параметры и количества записаны. Нет нормализации каждого патча. Это не точное воспроизведение нормированной Laplacian-пирамиды ProGAN; не сравнивайте напрямую с её числами. |
| `spatial` | Wasserstein между расстояниями до ближайших центроидов: внутри класса и направленно между каждой парой классов. Включает граничные компоненты, без edge correction; зависит от field of view. Для мембран это расстояние **между центроидами**, не от органеллы до поверхности мембраны. Изображение с <2 объектами не даёт внутриклассовой дистанции. |
| `joint_mmd` | **Наш вариант совместного MMD**, не FRD. По изображению: mean/std яркости; для каждого класса — coverage, mean/std foreground, число компонент/Mpx и средняя площадь компоненты. Пустой класс задаётся нулевыми coverage/count/appearance. Признаки стандартизуются по real; для постоянных real-признаков scale=1 с явным перечислением. RBF sigma=√D, biased MMD². Учитывает зависимости между свойствами, но сохраняет ограничения выбранных признаков. |

Если распределение пусто, расстояние неопределено: `null` или
`status: undefined_empty_distribution`. Недостаток наблюдений даёт
`insufficient_samples`. Это не идеальное совпадение. Верхний `status: complete`
означает завершение всех запрошенных вычислений, а не статистическую достаточность
каждой метрики: проверяйте также статусы внутри `metrics`.

## 6. JSON и воспроизводимость

Схема версии 2 содержит:

```text
schema_version
status                     complete / partial
metrics                    значения выбранных метрик
config                     фактический состав режима и параметры
encoders                   модели, ревизии, preprocessing, cache hit/miss
datasets.real/synthetic   пути, использованные изображения, размеры, пропуски
warnings                   ограничения конкретного запуска
versions                   версии библиотек
elapsed_seconds
```

Расчёт не выдаёт общий усреднённый балл: разные показатели имеют разные единицы
и измеряют разные свойства. Сравнивайте эксперименты при одинаковых моделях,
версиях, классах, параметрах выборки, шкале и пороге масок.

Физический размер пикселя должен быть сопоставим. CLI не читает nm/pixel и
не выполняет физически согласованный ресемплинг; нейросетям целые изображения
растягиваются до входного квадрата. Если исходные поля зрения сильно различны,
сначала подготовьте патчи одинакового физического размера. Режим патчинга
внутри SWD относится только к SWD и не меняет вход FID/KID/CMMD.

Рекомендуемый протокол: real-vs-real baseline на непересекающихся областях;
затем сравнение вариантов генератора; проверка реакции на известные изменения
шума, размытия, количества/размера органелл; финальная проверка DICE на
фиксированном реальном тесте. Соседние срезы и патчи одного объёма коррелированы.
Много патчей из трёх срезов не создают много независимых образцов.

## 7. Тесты и источники

```bash
# Контрольные тесты не скачивают веса и работают с requirements-lite.txt:
python -m unittest discover -s evaluation -t . -v
```

Проверяются формулы против прямых численных эталонов, недостаточные выборки,
отрицательный KID, PRDC, SWD, Vendi, режимы, явные пропуски моделей, строгий JSON,
сброс/повреждение кеша, форматы Lucchi/Kasthuri и геометрия масок.
Эти тесты не являются проверкой корреляции метрик с DICE.

- [FID: Heusel et al.](https://arxiv.org/abs/1706.08500), [pytorch-fid](https://github.com/mseitzer/pytorch-fid).
- [KID: Bińkowski et al.](https://arxiv.org/abs/1801.01401).
- [PRDC: Naeem et al.](https://arxiv.org/abs/2002.09797), [референсный код](https://github.com/clovaai/generative-evaluation-prdc).
- [CMMD: Jayasumana et al.](https://arxiv.org/abs/2401.09603), [референсный код](https://github.com/google-research/google-research/tree/master/cmmd).
- [Vendi Score](https://arxiv.org/abs/2210.02410).
- [Многомасштабный SWD в ProGAN](https://arxiv.org/abs/1710.10196) — источник идеи, не идентичный протокол.
- [EMcopilot: применение метрик изображений, геометрии и расположения объектов](https://www.nature.com/articles/s41524-025-01800-5).
- [GLCM/LBP](https://scikit-image.org/docs/stable/api/skimage.feature.html), [regionprops](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).

### Проверка базовой реализации (2026-09-30)

- 28 тестов прошли в CPU-окружении с полными зависимостями; тест адаптера CLIP
  использует маленькую случайно инициализированную модель только как fixture,
  её признаки никогда не используются в пользовательских отчётах.
- Basic с официальными весами FID проверен на 3 реальных и 3 синтетических
  изображениях Kasthuri, включая повторный offline-запуск из кеша.
- Medium проверен на 6 реальных и 6 синтетических изображениях Lucchi.
- Все метрики full, кроме CMMD, проверены на 3+3 изображениях Kasthuri
  с `--prdc-k 2` специально для маленькой контрольной выборки.
- Для CMMD проверены численная формула, preprocessing, батчирование и
  нормировка признаков. Полный прогон с pretrained CLIP пока не проверен:
  большая загрузка весов в проверочной среде была остановлена.
- CUDA и корреляция метрик с тестовым DICE не проверялись.

### Проверка task-метрик (2026-10-06)

- 42 теста (`test_evaluation`, `test_advanced`, `test_task_metrics`) прошли на CPU.
- Предоставленный legacy `Tiny_unet_v3` загрузился; forward адаптера совпал
  с исходным классом framework побитно на случайном batch 2×1×256×256.
- Все семь новых групп метрик прошли на 8 real + 8 synthetic изображениях
  с пользовательским checkpoint. Это технический smoke test: конфигурация
  checkpoint указывает обучение на синтетике, а не real-only.
- CUDA и связь proxy с downstream Dice этим запуском не проверены.

### Визуальное сравнение гистограмм: `--metrics histogram`

```bash
cd /mnt/d/course_work/data/synth_rep/Synthetics
python -m pip install -r evaluation/requirements-lite.txt
python -m evaluation \
  '/mnt/d/course_work/data/SLICES/cutting data 30 slices' \
  dataset/test_dataset_no_noise_lucchi \
  --metrics histogram --run-info intensity_histograms
```

Для каждого изображения из `original/` вычисляется гистограмма яркости на
256 общих интервалах `[0, --intensity-max]` (по умолчанию 255), нормируется
на количество его пикселей, затем усредняется по изображениям отдельно для
real и synthetic. Каждое изображение имеет одинаковый вес независимо от размера.
Без `--max-images` используются **все изображения** обеих директорий, включая
вложенные каталоги. С `--max-images N` используется выбранная по seed подвыборка.
Для 16-битных изображений задайте `--intensity-max 65535`. Индивидуальное
растяжение контраста не применяется.

Результаты:

- `results/report/<run>.json`: `metrics.histogram` содержит средние вероятности,
  границы интервалов, количество изображений, расстояния и `overlap` (сумма
  минимумов вероятностей; 1 означает совпадение распределений).
- `results/segmentation/<run>/plots/histogram.png` и `histogram.svg`: средние
  гистограммы real/synthetic и их накопленные распределения на одном рисунке.
- `results/segmentation/<run>/command.txt`: команда воспроизведения запуска.

Пути `metrics.histogram.plots` указаны относительно `artifacts.results_dir`.
Графики сохраняются без открытия окна; JSON продолжает выводиться в stdout.
Гистограммы не проверяют пространственное расположение органелл. Расстояние
Wasserstein рассчитано для яркости, нормированной на `[0,1]`.
В отличие от существующей `intensity`, здесь усредняются нормированные
гистограммы изображений, а не объединяются все пиксели.

Метрика дополнительная: включается через `--metrics histogram` или `--metrics all`,
пресеты basic/medium/full не меняются. Для совместного запуска используйте
`--metrics frozen_seg histogram` с теми же `--task-checkpoint`, `--task-adapter`
и `--task-config`, что и для обычного frozen_seg. Явный список `--metrics`
заменяет пресет. Для histogram отдельно модель и маски не требуются.


### Гистограммы яркости по классам

`--metrics histogram` автоматически добавляет сравнение каждого общего класса
из папок масок. Для выбора классов добавьте, например, `--classes mitochondria vesicles`.
Модель не требуется: используются исходные маски датасетов, а не предсказания frozen_seg.
Если общих классов нет, сохраняется только общий график с предупреждением в JSON.

Для каждого класса учитываются яркости изображения **только внутри маски**
(`mask > --mask-threshold`, по умолчанию 0). Гистограмма каждого непустого
изображения нормируется на число пикселей класса и затем усредняется с равными
весами. Пустые маски исключаются из среднего; число таких изображений и число
изображений с классом записываются отдельно. Это распределение яркости внутри
органеллы, а не частота встречаемости органелл. Перекрывающиеся маски учитываются
независимо. Без `--max-images` обрабатываются все изображения.

Графики PNG/SVG: `results/segmentation/<run>/plots/classes/<class>/histogram.*`.
Числа, средние распределения и пути: `metrics.histogram.classes.<class>` в JSON.
При отсутствии положительных пикселей в одном домене расстояния и overlap равны
`null`, статус — `undefined_empty_class`; на графике показано отсутствие класса.
Пропущенная маска по умолчанию вызывает ошибку; `--allow-missing-masks` разрешает
пропуск с отдельным счётчиком. Пустая маска и отсутствующий файл различаются.

```bash
python -m evaluation REAL_DIR SYNTHETIC_DIR --metrics histogram
# Только выбранные классы (общая гистограмма также сохраняется):
python -m evaluation REAL_DIR SYNTHETIC_DIR --metrics histogram --classes mitochondria vesicles
```
