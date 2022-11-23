#! /bin/bash

# https://stackoverflow.com/a/9103113/6390175
git submodule foreach git pull -q origin main
