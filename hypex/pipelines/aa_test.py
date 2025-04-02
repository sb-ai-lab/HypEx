from hypex.pipeline import Pipeline


class AATest(Pipeline):
    """Pipeline проведения деления
    здесь будет полное описание как с этим работать
    """

    def custom_run(self) -> 'AATest':
        print("AATest processing...")
        print(self.context.pipeline_context['kwargs'])

        return self