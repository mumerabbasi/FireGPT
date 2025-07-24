#!/usr/bin/env bash

ollama serve &
ollama list
ollama pull qwen2.5vl

ollama serve &
ollama list
ollama pull qwen3:8b