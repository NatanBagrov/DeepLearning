#!/bin/bash

rsync -rav -e ssh --include="*/" --include='*.png' --exclude="*" rishon:DL/graphs/ models/graphs