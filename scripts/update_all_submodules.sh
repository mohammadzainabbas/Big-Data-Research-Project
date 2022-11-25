#! /bin/bash

# https://stackoverflow.com/questions/4604486/how-do-i-move-an-existing-git-submodule-within-a-git-repository for moving submodules
# https://stackoverflow.com/a/9103113/6390175
git submodule foreach git pull -q origin main
