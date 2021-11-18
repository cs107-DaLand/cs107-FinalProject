#!/usr/bin/env bash

set -e
coverage run -m pytest test_forward.py -v
coverage report --fail-under=90