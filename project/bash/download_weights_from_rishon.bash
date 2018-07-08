#!/bin/bash

rsync -rav -e ssh --include="*/" --include='*.h5' --exclude="*" rishon:DL/project/saved_weights/ models/saved_weights
