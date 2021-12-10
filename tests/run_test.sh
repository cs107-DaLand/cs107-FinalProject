#!/usr/bin/env bash
export PYTHONPATH='../src/'
set -e
coverage run -m pytest test_forward.py test_optimize.py -v --disable-pytest-warnings
coverage report --fail-under=90
coverage html
