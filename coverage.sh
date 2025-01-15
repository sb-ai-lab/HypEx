#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

coverage run --include="hypex/dataset/*" unitests/unitests.py

coverage report -m

coverage html -d unitests/coverage_report

rm -f .coverage