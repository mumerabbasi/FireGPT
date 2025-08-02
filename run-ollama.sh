#!/usr/bin/env bash

ollama serve &
ollama list
ollama pull qwen2.5vl qwen3:8b

