#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

# Запуск покрытия тестов
coverage run --include="hypex/dataset/*" unitests/unitests.py

# Печать отчета о покрытии в консоль
coverage report -m

# Генерация HTML отчета
coverage html -d unitests/coverage_report

rm -f .coverage