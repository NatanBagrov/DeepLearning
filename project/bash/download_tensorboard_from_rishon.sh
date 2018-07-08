#!/bin/bash

rsync -rav -e ssh --include="*/" rishon:DL/project/logs/ models/logs

