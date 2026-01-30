# Fluoddity-Core

A minimal, teaching-focused rebuild of the SimScratch simulation engine core.

## Purpose

This repository contains a from-scratch rebuild of SimScratch's core engine with:
- Bare minimum feature set
- Clean, educational code structure
- Clear documentation for understanding key concepts

## Structure

  - Base Python files
  - `shaders/` - GLSL shader files (.frag, .vert, .glsl)

## Virtual Environment

Activate with:
```
& C:\Users\jgeld\Documents\KodeLife\Fluoddity-Core\Scratch.venv\Scripts\Activate.ps1
```

## Development Approach

This is a "single blind" rewrite where features are built from requirements rather than
by copying the original implementation. Reference code is consulted only when explicitly
needed for specific components. The user will tell you if you should access any reference files.
Make things as simple as possible, but no simpler.

## Hot Reload Requirements

This is a teaching tool where users will frequently tinker with shaders and see results update
immediately without restarting the program:

1. **Isolated shader initialization** - Use helper methods for shader/buffer setup so they can be
   re-initialized mid-stream
2. **Graceful compilation failures** - Failed shader compilation must not crash the program. Log
   errors to console and keep the previous working program
3. **Safe uniform setting** - Use `tryset()` from `gl_utils.py` for all uniforms.
   Uniforms may be optimized out when shaders are modified, and ModernGL throws errors for
   missing uniforms

