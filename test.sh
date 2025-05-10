#! /bin/sh

export PYTHONPATH="$(dirname $0)"
python3 -m unittest discover -p '*test.py'
