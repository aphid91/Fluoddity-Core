# Hive-Core

A minimal, teaching-focused rebuild of the SimScratch simulation engine core.

## Purpose

This repository contains a from-scratch rebuild of SimScratch's core engine with:
- Bare minimum feature set
- Clean, educational code structure
- Clear documentation for understanding key concepts

## Structure

- `reference/` - Source files from original SimScratch for selective reference
  - Base Python files
  - `scripts/` - Migration and utility scripts
  - `services/` - Core services (arrow debug, config saver, entity picker, etc.)
  - `shaders/` - GLSL shader files (.frag, .vert, .glsl)
  - `state/` - State management modules
  - `utilities/` - Helper utilities (GL helpers, frame assembly, recording, etc.)

## Virtual Environment

Activate with:
```
& C:\Users\jgeld\Documents\KodeLife\Hive-Core\Scratch.venv\Scripts\Activate.ps1
```

## Development Approach

This is a "single blind" rewrite where features are built from requirements rather than
by copying the original implementation. Reference code is consulted only when explicitly
needed for specific components. The user will tell you if you should access any reference files.
Make things as simple as possible, but no simpler.

