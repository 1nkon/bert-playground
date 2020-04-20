#!/bin/bash


curl -d "{\"sentence\": \"$1\",\"simple\": true}" -H 'Content-Type: application/json' http://localhost:5000

