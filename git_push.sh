#!/bin/bash


# git commands
# ------------

git init
git add .
git commit -m "$1"
git branch -M main
git remote add origin "$2"
git push -u origin main
