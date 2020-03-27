#!/bin/bash

FWDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
WORK_DIR="$FWDIR/.."

export PYTHONPATH="$WORK_DIR/python/:$PYTHONPATH"

executable=$1

python "$executable" "${@:2}"
