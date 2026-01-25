#!/bin/bash

if [ "$1" == "" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 AGENT_NAME"
    echo "Run a loop taking and working on tasks from https://github.com/users/yieldthought/projects/6. AGENT_NAME should be unique to this worker."
    exit 1
fi
while true; do codexapi task -p https://github.com/users/yieldthought/projects/6 -n "$1" tasks/*; sleep 60; done
