#!/usr/bin/env bash
export PYTHONPATH='../src/'
set -e
# pytest --cov=src ./
coverage run -m pytest test_forward.py -v