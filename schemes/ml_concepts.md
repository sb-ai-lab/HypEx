# ML Pipeline Concepts

Концептуальная модель ML-поддержки в HypEx.

## Обзор

ML pipeline отличается от статистических Executor'ов наличием **состояния** после обучения. Ключевое решение: состояние хранится в данных, а не в Executor'ах.

## Новые абстракции

### MLExperimentData

Наследник `ExperimentData` с расширениями для ML:

```mermaid
classDiagram
    ExperimentData <|-- MLExperimentData

    class ExperimentData {
        +ds: Dataset
        +additional_fields: Dataset
        +analysis_tables: dict
        +variables: dict
    }

    class MLExperimentData {
        +pipeline: list~str~
        +artifacts: dict~str, Any~
        +save(path)
        +load(path)
    }
```

**Поля:**
- `pipeline: list[str]` — упорядоченный список ID выполненных Executor'ов
- `artifacts: dict[str, Any]` — fitted объекты по ID Executor'а

**Особенности:**
- Создаётся как **копия** исходных данных
- **Мутабельный** внутри pipeline (для эффективности)
- Поддерживает **сохранение/загрузку** с диска

**Пример:**
```python
data.pipeline = ['NaFiller╤aaa╤', 'Scaler╤bbb╤', 'RandomForest╤ccc╤']

data.artifacts = {
    'Scaler╤bbb╤': fitted_scaler,
    'RandomForest╤ccc╤': fitted_model,
    # NaFiller нет — он stateless
}
```

### MLExecutor

Базовый класс для ML Executor'ов. Наследует от Executor.

```mermaid
classDiagram
    Executor <|-- MLExecutor
    MLExecutor <|-- MLTransformer
    MLExecutor <|-- MLPredictor

    class Executor {
        +id: str
        +execute(data) ExperimentData
    }

    class MLExecutor {
        +mode: str
        +execute(data) MLExperimentData
    }

    class MLTransformer {
        mode: fit | transform | fit_transform
    }

    class MLPredictor {
        mode: fit | predict | fit_predict
    }
```

**Параметры:**
- `mode` — режим работы, задаётся при создании

**Поведение execute:**
- Сохраняет артефакт в `data.artifacts[self.id]` (если есть)
- Регистрирует себя в `data.pipeline`

### MLTransformer

Executor для преобразования данных (scaler, encoder и т.д.).

**Режимы:**
- `fit` — обучение, сохраняет артефакт
- `transform` — применение, читает артефакт
- `fit_transform` — обучение + применение

### MLPredictor

Executor для моделей (классификация, регрессия и т.д.).

**Режимы:**
- `fit` — обучение, сохраняет артефакт
- `predict` — предсказание, читает артефакт
- `fit_predict` — обучение + предсказание

## Поток данных

```mermaid
flowchart LR
    subgraph Input
        ED[ExperimentData]
    end

    subgraph ML Pipeline
        MED["MLExperimentData<br/>копия, мутабельный"]
        T1["MLTransformer<br/>mode=fit_transform"]
        T2["MLTransformer<br/>mode=fit_transform"]
        P["MLPredictor<br/>mode=fit_predict"]
    end

    subgraph Output
        Result["MLExperimentData<br/>+ pipeline<br/>+ artifacts"]
    end

    ED -->|copy| MED
    MED --> T1
    T1 --> T2
    T2 --> P
    P --> Result
```

## Восстановление Pipeline

Pipeline можно восстановить из MLExperimentData:

1. Из `pipeline` получаем порядок ID
2. По ID восстанавливаем Executor'ы (класс + параметры)
3. Из `artifacts` достаём fitted состояние
4. Применяем к новым данным с `mode=transform/predict`

```mermaid
flowchart LR
    subgraph Saved
        MED1[MLExperimentData]
    end

    subgraph Restored
        E1["Executor 1<br/>mode=transform"]
        E2["Executor 2<br/>mode=transform"]
        E3["Executor 3<br/>mode=predict"]
    end

    subgraph New Data
        ND[New ExperimentData]
        NR[Predictions]
    end

    MED1 -->|restore| E1
    MED1 -->|restore| E2
    MED1 -->|restore| E3
    MED1 -->|artifacts| E1
    MED1 -->|artifacts| E2
    MED1 -->|artifacts| E3

    ND --> E1 --> E2 --> E3 --> NR
```

## Связь с существующей архитектурой

| Существующее | ML-расширение |
|--------------|---------------|
| ExperimentData | MLExperimentData |
| Executor | MLExecutor |
| Calculator | MLTransformer, MLPredictor |
| analysis_tables | artifacts |
| — | pipeline |

Принципы сохраняются:
- Единый интерфейс `execute(data) → data`
- Идентификация по ID
- Результаты по ID Executor'а
- Композируемость
