#!/usr/bin/env bash

ollama serve & sleep 20
ollama pull qwen2.5vl
ollama pull qwen3:8b

