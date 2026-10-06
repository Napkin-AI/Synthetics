# Проверка EM-синтетики замороженным сегментатором

## Что именно проверяется

Обучаем сегментатор **F_R только на оригинальных данных**, выбираем checkpoint
и пороги на отдельной real validation части. После этого фиксируем параметры,
BatchNorm, preprocessing, порядок классов и пороги. Ни обучение, ни подбор
checkpoint, ни адаптация BatchNorm на синтетике не допускаются.

Для каждого варианта генератора S_j вычисляем:

```
Q_real = macro Dice / IoU(F_R(R_reference), Y_reference)
Q_synth_j = macro Dice / IoU(F_R(S_j), Y_synthetic_j)
gap_j = Q_synth_j - Q_real
```

Это **перенос сегментации real → synthetic**, а не доказательство равенства
распределений и не измерение пользы синтетики для обучения. Хороший Q_synth
означает, что модель умеет распознавать размеченные структуры синтетики.
Слишком простая синтетика может иметь Q_synth выше настоящих изображений.
Систематические ошибки масок генератора тоже влияют на результат.
Уверенность и entropy без сравнения с масками не являются оценкой качества.

## Добавленные метрики

| CLI имя | Содержание | Направление |
|---|---|---|
| `em_kid` | Несмещённая оценка polynomial MMD² `(x·y/d+1)^3` в признаках пользовательского encoder, global и по классам | Меньше ближе; отрицательная конечновыборочная оценка допустима |
| `em_fd` | Fréchet distance в тех же признаках | Меньше ближе; при N ≤ D ковариация вырождена |
| `c2st` | Различение домена L2 logistic classifier на отложенных группах, ROC-AUC и balanced accuracy | AUC около 0,5; `macro_available` усредняет `abs(AUC_c-0.5)` |
| `frozen_seg` | Dice, IoU, precision, recall, boundary F1 по классам, Brier и binary entropy, per-image результаты, synthetic-minus-real gap | Высокая точность; gap интерпретировать вместе с трудностью сцен |
| `shape` | Wasserstein по площади, периметру, осям, eccentricity, solidity, circularity, толщине, скелету, Euler number, доле и плотности объектов | Меньше ближе |
| `psd` | Расстояние radial log-power кривых + angular PSD distribution | Меньше ближе |
| `noise` | Статистики, histogram и lag-1 autocorrelation остатка `I − Gaussian(I)` | Меньше ближе |

`shape/psd/noise` входят в `medium/full`. `basic` по-прежнему FID/KID.
Метрики с пользовательской моделью можно выбрать через `--metrics`.
При `--task-checkpoint` без явного `--metrics` к режиму автоматически добавляется
`frozen_seg`; `--exclude-metrics frozen_seg` отменяет это добавление.
Автоматически подобрать корректный checkpoint невозможно. `--metrics all`
включает и их. Без checkpoint `--skip-unavailable` отмечает их как unavailable.
Ошибочный порядок каналов, группировка или размер выхода — ошибки, а не
отсутствующие зависимости; они не подавляются этим флагом.

### Детали, влияющие на интерпретацию

- Encoder получает **только изображение**, не маски. Для class-conditional
  признаков маска после inference уменьшается через area interpolation до
  разрешения feature map и задаёт веса spatial mean. Один image/class — один
  вектор. Это не независимые пиксели и не object-crop KID. Маленькие органеллы
  могут плохо представляться на слишком грубом feature map; выбирайте слой
  заранее, например `down2`, и фиксируйте его для всех вариантов.
- Маски меньше `--min-class-pixels 16` пропускаются при feature pooling.
  Возвращаются числа наблюдений и списки image keys. Классы с недостаточными
  данными имеют null и статус. Macro сравним только для одного и того же набора
  доступных классов. Доли/частоты отсутствующих классов проверяйте `coverage`.
- EM-KID/FD используют исходный масштаб activations. Не сравнивайте значения
  разных checkpoint/layers и Inception. Нужен real-real ориентир в том же
  пространстве. KID std — разброс подвыборок, не доверительный интервал.
- Для Dice/IoU сначала суммируются TP/FP/FN по набору для каждого класса,
  затем считается macro по foreground-классам с определённым результатом.
  Если target и prediction класса оба пусты, результат null, не 1. Если
  target пуст, но prediction непуст — Dice/IoU равны 0. В JSON также есть
  mean image scores, support и per-image TP/FP/FN.
- В `multilabel` каналы независимы: sigmoid и фиксированный порог, по умолчанию
  0,5. Пересекающиеся маски допустимы. `multiclass` использует softmax/argmax,
  требует канал `background` и все foreground-классы; пересечения и пропущенные
  маски отклоняются. Background исключён из macro.
- Boundary F1: доли граничных пикселей, имеющих соответствие в радиусе
  `--boundary-tolerance 2` pixels, затем harmonic mean. Это boundary pixel F1,
  **не** surface-area-weighted Surface Dice. Масштаб nm/pixel должен совпадать.
- `shape`: 8-connected components; объекты, обрезанные краем, исключаются из
  object geometry и thickness. Density/coverage/Euler учитывают всю маску.
  Толщина — `2*EDT` на medial axis (растровое приближение), длина скелета —
  число его пикселей. Масштаб Wasserstein: real IQR → real std → 1 в исходных
  единицах, если real feature постоянен. Каждый fallback записывается.
  Macro этих расстояний — диагностическое среднее с равными весами,
  **не откалиброванный прогноз Dice**; пропущенные object features не равны нулю.
- `psd`: Hann window, power/sum(window²), частоты 0–0,5 cycles/pixel, DC
  исключён. Есть абсолютный и центрированный radial log1p-power L1, а также
  angular JS/Hellinger/TV. `noise` содержит края и текстуры, а не только шум
  микроскопа; его sigma фиксируется через `--noise-sigma`.
- Пользовательская модель не скачивается. В текущей версии task inference
  повторяется при каждом запуске; Inception/CLIP кеш остаётся отдельным.
  Выбирайте сразу все нужные task метрики: они используют один проход модели.

## Запуск с вашим UnetClass

В `examples/unetclass.json` записаны framework_root, training_config и
feature_layer для предоставленного checkpoint. **Этот пример не real-only**:
в соседней конфигурации оригиналам назначена доля 0, синтетике 1.0.
`DataGeneratorReaderAll` при доле 0 выбирает 0 изображений. Адаптер записывает
`training_provenance: synthetic_only` и источники из конфигурации в отчёт.
Это свидетельство из конфигурации, а не независимый аудит истории весов.

Из корня Synthetics, техническая проверка на небольшом наборе:

```bash
python -m evaluation \
  '/mnt/d/course_work/data/SLICES/cutting data 5 slices' \
  dataset/test_dataset_no_noise_lucchi \
  --metrics em_kid em_fd c2st frozen_seg shape psd noise \
  --task-checkpoint '/mnt/d/course_work/4_course/expertiments/7_sem/0_slices_exp_2_upd/segmentation/2025_11_27_13_54_42_1.0/model_by_config_test_&_tiny_unet_v3_!.pt' \
  --task-adapter evaluation.unetclass_adapter:load \
  --task-config evaluation/examples/unetclass.json \
  --model-classes mitochondria PSD vesicles axon boundaries 'mitochondrial boundaries' \
  --group-pattern '^(.*?)(?:\.png_.*)?$' \
  --max-images 8 --batch-size 2 --device cpu --run-info synthetic_trained_smoke
```

Адаптер поддерживает именно legacy full-module `Tiny_unet_v3`. Он использует
`model_block.py` исходного framework и совместимый forward, не требует SMP и
модулей обучения. Full-module pickle следует загружать только из доверенного
источника. Для переносимости новых моделей предпочтителен собственный адаптер
со `state_dict` и `torch.load(..., weights_only=True)`.

Порядок шести каналов проверяется по training_config. Нормализация: grayscale
float32 / `--intensity-max` (255), без resize и аугментаций. Размеры должны
делиться на 16. Для full-size срезов заранее сделайте одинаковые по физическому
масштабу patches; adapter не выполняет sliding-window inference.

Для настоящего эксперимента замените checkpoint и training_config на real-only,
передайте отдельный `REAL_REFERENCE`, уберите ограничение 8 изображений и
добавьте `--require-real-only`. Этот флаг проверяет **записанное происхождение**,
но не может доказать отсутствие утечки в неизвестной истории обучения.

### Новый real-only checkpoint и результаты на диске

Для каталога `10_slices_exp_2/segmentation/2025_05_16_00_45_13_0_0` сохранённая
конфигурация содержит real proportion=1, synthetic proportion=0. Добавлен
готовый `examples/unetclass_real_only.json`. Команда из корня Synthetics:

```bash
python -m evaluation /path/to/REAL_REFERENCE /path/to/SYNTHETIC \
  --metrics frozen_seg em_kid em_fd shape psd noise \
  --task-checkpoint '/mnt/d/course_work/4_course/expertiments/7_sem/10_slices_exp_2/segmentation/2025_05_16_00_45_13_0_0/model_by_config_test_1_&_tiny_unet_v3_!.pt' \
  --task-adapter evaluation.unetclass_adapter:load \
  --task-config evaluation/examples/unetclass_real_only.json \
  --model-classes mitochondria PSD vesicles axon boundaries 'mitochondrial boundaries' \
  --require-real-only --run-info real10_vs_lucchi --batch-size 4
```

`REAL_REFERENCE` должен исключать обучающие source slices. В локальной копии
`cutting data 10 slices` это `training0013`, `0028`, `0046`, `0061`, `0091`,
`0105`, `0119`, `0134`, `0135`, `0164` (для всех действует префикс `training`).
Проверка имён не заменяет пространственный buffer: соседние срезы коррелируют.

Все CLI-результаты автоматически записываются в
`evaluation/results/report/<date_config_info>.json`. Маски лежат в
`evaluation/results/segmentation/<date_config_info>/{real,synthetic}/`:
`predictions/<class>/<image_key>.png` и `targets/<class>/<image_key>.png`.
Рядом в `original/` сохраняются побайтовые копии исходных изображений в их
формате и разрядности; пути и SHA256 указаны в `per_image[].original`.
Это точные бинарные массивы из расчёта Dice/IoU, сохранённые в 0/255.
В папке запуска также находятся `command.txt` и `run.json` со статусом и
параметрами. Привязка исходных изображений и масок к файлам результата —
в `frozen_seg.<domain>.per_image`. При чтении PNG foreground определяется `>0`.
Проверить Dice можно повторным суммированием TP/FP/FN по сохранённым парам,
с последующим усреднением определённых foreground class Dice, как в отчёте.

Путь меняется через `--results-dir`, суффикс имени — через `--run-info`.
Вероятности и logits не сохраняются: PNG достаточно для повторного Dice/IoU,
но для пересчёта entropy/Brier или нового порога потребуется inference.

Без `--max-images` обрабатываются все поддерживаемые изображения обоих наборов.
`--batch-size` по умолчанию 16; последний неполный batch тоже обрабатывается.
Флага `--max-batch` нет. Уменьшение batch-size уменьшает расход памяти, не размер
набора. Команда без ограничения выборки есть в
[README: полный запуск](README.md#полный-запуск-с-real-only-u-net).

### Делать ли require-real-only обязательным?

В общем CLI default остаётся **false**, флаг включается явно. Это позволяет
исследовать также synthetic-only и mixed models, с явным происхождением в JSON.
Для вашего основного эксперимента **real → synthetic включайте флаг всегда**.
Если provenance неизвестен, строгий режим тоже отклонит checkpoint.

Модель, обученная на синтетике, может показывать завышенные показатели как
оценка независимого переноса с real; особенно если тест содержит те же сцены
или генераторные seeds. Рост не гарантирован математически и касается прежде
всего frozen_seg и зависимых от этого encoder метрик, а не автоматически всех
статистик изображений. Даже real-only модель может иметь утечку, если реальные
тестовые patches пересекаются с обучающими. Флаг проверяет provenance по
сохранённой конфигурации, но не способен восстановить всю историю обучения.

## Группировка для C2ST

Нельзя случайно распределять overlapping patches одного среза по train/test.
Передайте `--groups-json groups.json`:

```json
{
  "real": {
    "training0028.png_256_64_0": "volume1_z0028",
    "training0028.png_256_64_1": "volume1_z0028"
  },
  "synthetic": {
    "0_2026_01_24": "scene0",
    "1_2026_01_24": "scene1"
  }
}
```

Ключ — относительный путь изображения без последнего расширения с учётом
Lucchi naming alias из `data.image_key`; в manifest должны быть **все выбранные
изображения**, JSON выше показывает только формат. Для разных вариантов одного
синтетического scene ID задавайте одну группу. Группы domain-local; если real
и synthetic сделаны из одних исходных сцен, этот протокол не обеспечивает
разделение общей сцены между доменами — используйте независимые reference scenes.

Альтернатива — `--group-pattern` с одной capture group. Пример выше объединяет
`training0028.png_256_64_*` в `training0028`, оставляя уникальные synthetic scene
имена отдельными группами. Для соседних срезов одного volume лучше задавать
более крупные пространственные блоки через JSON.

По умолчанию 3 folds, минимум 3 группы **в каждом домене для каждого класса**.
Если их мало, C2ST этого класса получает `insufficient_groups`. StandardScaler
и classifier обучаются только на train folds; test groups/indices записываются.
Среднее fold AUC около 0,5 не доказывает равенства распределений; classifier
линейный, нелинейные отличия он может не обнаружить. P-value не вычисляется.

## Интерфейс для другой модели

```python
# my_adapter.py — этот файл и код архитектуры доступны при запуске.
def load(checkpoint, device, config):
    import torch
    from my_project import SegmentationModel

    model = SegmentationModel(**config["architecture"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            # x: B,1,H,W float32 [0,1]. Здесь воспроизведите TRAIN preprocessing.
            # Реализуйте вызовы под API своей архитектуры:
            features = self.model.encode(x)       # spatial B,D,Hf,Wf
            logits = self.model.decode(features)  # B,C,H,W, ДО sigmoid/softmax
            return {"features": features, "logits": logits}

    return Adapter()
```

Запуск: `--task-adapter /path/my_adapter.py:load --task-checkpoint model.pt
--task-config config.json`. В config можно указать `training_provenance`:
`real_only`, `mixed`, `synthetic_only` или `unknown`; это декларация владельца
модели. Adaptor metadata имеет приоритет, поэтому нельзя переопределить
обнаруженное synthetic_only в UnetClass простым полем config.

Если нужны только feature метрики, достаточно `features`; если только
`frozen_seg`, достаточно `logits`. Выходы должны быть конечными. Размер logits
должен совпадать с исходным изображением: padding/resize обратно делает адаптер.
Для TorchScript dictionary model `--task-adapter` можно не задавать.
Runner вызывает eval(), отключает gradients и использует inference_mode.

## Практический протокол эксперимента

1. Разделите **до нарезки patches** оригинальные volumes/пространственные блоки
   на R_train, R_val, R_reference и R_test. Соседние срезы и overlapping patches
   оставляйте в одном блоке; между z-блоками оставьте буфер, согласованный с
   пространственной корреляцией данных. R_test до финального выбора закрыт.
   Пять исходных срезов не заменяют несколько независимых volumes.
2. Обучите F_R на R_train, checkpoint/пороги выберите на R_val. В существующем
   framework недостаточно поменять доли на real=1/synthetic=0: стандартный
   `share_validat=0.2` делит patches и может давать утечку. Обеспечьте внешнее
   разделение по блокам и отдельный validation loader. Нулевую synthetic запись
   лучше удалить из data_info целиком. Сохраните списки файлов/групп, seed,
   preprocessing, канал→класс и SHA256 checkpoint.
3. Проверьте Dice/IoU F_R на R_reference по каждому классу. Если F_R плохо
   сегментирует реальные PSD/vesicles, его оценки этих классов синтетики пока
   малоинформативны. Проверяйте одинаковые правила разметки, nm/pixel и FOV.
4. Зафиксируйте encoder/layer и reference. Сначала получите real-real floor
   для `em_kid/em_fd/shape/psd/noise`: сравнивайте непересекающиеся блоки R_reference
   с одинаковыми размерами выборок. Не сравнивайте массив с самим собой как
   эмпирический floor. Пространственная корреляция сохраняется и между срезами.
5. Подготовьте 12–20 конфигураций S_j, по возможности по 500–2000 изображений.
   Контролируемые оси: noise, blur, contrast, membrane thickness, organelle size,
   density. Используйте одинаковые scene seeds между вариантами и отдельные
   seeds для финального synthetic holdout. Включите baseline, no-noise и явно
   испорченные варианты; параметры изменяйте по одной оси.
6. Одной замороженной F_R посчитайте все метрики для каждого S_j и одного
   R_reference. Сохраняйте per-class scores/support. Synthetic masks используются
   как GT **только при оценке**, не подаются модели. Пороги не подбирайте на S_j.
7. Оценивайте неопределённость Dice/IoU bootstrap по source groups, суммируя
   per-image TP/FP/FN внутри выбранных групп и затем пересчитывая macro.
   При сравнении вариантов с одинаковыми scene seeds используйте paired
   group bootstrap. Pixel/patch bootstrap при коррелированных данных неверен.
8. Для проверки **полезности для обучения** возьмите 4–8 конфигураций по всему
   диапазону proxy и 2–3 training seeds. Обучите новые одинаковые сегментаторы:
   (a) real only, (b) real + S_j, при одинаковом бюджете optimizer steps,
   real exposures и заранее фиксированном mixing protocol. Контроль дополняйте
   повторными real samples/augmentation, чтобы отделить эффект synthetic от
   числа шагов. При интересе к synthetic-only переносу добавьте (c) S_j only.
   На закрытом R_test измерьте `ΔDice = Dice(real+S_j) − Dice(real only)`.
9. Ранжируйте конфигурации по Spearman между proxy и ΔDice/ΔIoU. Единица
   наблюдения — **конфигурация**, seeds усредняйте внутри неё; bootstrap по
   конфигурациям. Для множества метрик учитывайте FDR. Проверяйте ранжирование
   на отложенных конфигурациях: нельзя оценивать и калибровать веса на одних
   и тех же presets. Не присваивайте заранее веса единому quality score.

Таким образом дорогой training нужен для ограниченного числа контрольных
конфигураций. Для обычного подбора параметров генератора остаётся inference
одной F_R и вычисление распределительных метрик. Высокая корреляция с downstream
quality из deep-research отчёта пока является гипотезой, а не установленным
свойством EPFL или этого генератора.

## Первичные источники

- [KID / Demystifying MMD GANs](https://arxiv.org/abs/1801.01401): polynomial MMD
  estimator; замена Inception на пользовательский encoder — явно обозначенная
  модификация, значения разных представлений несопоставимы.
- [Revisiting Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545):
  held-out domain classification. Здесь реализован диагностический grouped
  logistic вариант, без заявления о формальном permutation test.
- [On Learning Invariant Representation for Domain Adaptation](https://arxiv.org/abs/1901.09453):
  совпадения marginal features недостаточно для гарантии переноса задачи.

## Проверки реализации

```bash
python -m unittest evaluation.test_evaluation evaluation.test_advanced \
  evaluation.test_task_metrics -v
```

Тесты покрывают численные формулы, empty classes, mask overlap, channel mapping,
source-group isolation, C2ST на разделённых/постоянных признаках, изменение
толщины/направления PSD и end-to-end запуск с toy encoder/segmentor.
Toy модель проверяет код, а не научную валидность proxy.
