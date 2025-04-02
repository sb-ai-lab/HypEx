from hypex.pipeline import Pipeline


class DataCleaner(Pipeline):
    """Pipeline для очистки данных"""

    def custom_run(self) -> 'DataCleaner':
        print("Cleaning data...")
        # Пример работы с данными
        self.context.df.dropna(inplace=True)
        #  Параметры находятся в контексте пайплайна
        kwargs = self.context.pipeline_context['kwargs']
        data = kwargs.get('data', None)
        print(data)
        self.context.pipeline_context['cleaned'] = True
        return self