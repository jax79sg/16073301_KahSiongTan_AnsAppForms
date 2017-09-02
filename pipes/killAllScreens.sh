#!/usr/bin/env bash
screen -ls | grep Detached | cut -d. -f1 | xargs kill