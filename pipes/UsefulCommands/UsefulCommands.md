#!/usr/bin/env bash

#Find top 100 sort files in current dir that do not have *.doc.txt in them
find . -type f ! -name *.doc.txt | sort | head -n 100

#Copy results of "Find top 100 sort files in current dir that do not have *.doc.txt in them" to temp folder
cp `find . -type f ! -name *.doc.txt | sort | head -n 100` temp