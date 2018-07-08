#!/bin/bash

rsync -rav -e ssh --include="*/" --include='*.py' --include='*.jpg' --include='*.JPEG' --include='*.bash'  --exclude="*" . rishon:DL/project

