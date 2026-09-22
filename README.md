# Fluoddity-Core
This is a stripped down version of:
https://github.com/aphid91/Fluoddity
Fluoddity-Core contains just enough machinery to load and run a config (no parameter sweeps; sensor distance is the only parameter that can jitter). It is meant as a companion to the full Fluoddity repo for those who want to tinker and/or understand the algorithm without digging through vibe-coded bells and whistles.
For more information see the Readme for Fluoddity

It also hosts this claude coded webgl port of the core engine: https://aphid91.github.io/Fluoddity-Core/ This demo can be found in the docs/ folder ("docs" folder is for github pages integration)

## Configs
Configs live in `physics_configs/`. Most of them are the `physics_configs/Core` set from the full Fluoddity repo, which is the subset that only uses parameters this engine implements. Two settings are read from the config beyond the raw physics parameters:

- `settings.initial_conditions` -- where each cohort starts when the simulation resets. 0=Grid (cohorts laid out on a grid filling the canvas), 1=Random (cohorts scattered across the canvas), 2=Ring (cohorts spaced evenly around a circle). Configs saved before this setting existed default to Grid.
- `jitters.SENSOR_DISTANCE` -- proportional randomness applied to `sensor_distance` per particle, per frame. 0.0 (the default) is off; 0.5 means each sensor reading is taken somewhere within +/-50% of the configured distance. The full Fluoddity build can jitter every physics parameter; this one carries only sensor distance because it is the parameter whose jitter changes the look the most.

## Algorithm Structure
System state consists of a particle buffer called "entities" and a texture that stores particle trails called "canvas". 
physics steps work like this:

### Entity Update
- Each particle in entities reads the canvas at a pair of sensor locations.
- The particle extracts the flow/current vector from each sensor reading
- "calculate_entity_behavior()" takes this information and processes it with constants taken from the .json config file (see entity_update.glsl comments for details on this process)
- calculate_entity_behavior outputs a vec2 force and vec2 strafe.
- we update particle state with: velocity =velocity*drag + force; and position += velocity + strafe;
### Brush Update
- In order to write new trails to the canvas, we must splat all the particles to their locations.
- A "brush" texture with the same dimensions as canvas acts as a staging area for these newly created trails.
- We use instanced rendering with one instance per entity and additive blending.
- Each particle draws a small gaussian kernel with color == (velocity_x, velocity_y, 0.01, 1) * kernel. (only the velocity terms are used currently, the 0.01 is mostly placeholder)
### Canvas Update
- The canvas update is a simple frag shader. Each frame, the trails diffuse and fade away, while we mix in the newly laid trails from brush.
- diffusion is handled by a simple 4 neighbor weighted average of the canvas
- trail fade is performed by mixing old trails (pre diffused) and new trails (from brush) with canvas_out = trail_persistence*canvas_in + (1-trail_persistence)*brush_in.
- This mix() style trail persistence ensures that the equilibrium trail intensity is independent of the specific trail-persistence value
