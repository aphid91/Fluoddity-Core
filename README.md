# Fluoddity-Core
This repo is a simplified version of:
https://github.com/aphid91/Fluoddity
Fluoddity-Core contains just enough machinery to load and run a Fluoddity config (no jitter or parameter sweeps). 
For more information see the Readme for Fluoddity

## Structure
System state consists of a particle buffer called "entities" and a texture that stores particle trails called "canvas". 
physics steps work like this:

### Entity Update
- Each particle in entities reads the canvas at a pair of sensor locations.
- The particle extracts the particle flow/current vector: canvas.xy, from each sensor
- The entity_update.glsl function "calculate_entity_behavior()" takes this information and processes it with constants taken from the .json config file 
- 
