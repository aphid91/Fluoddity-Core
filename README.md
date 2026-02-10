# Fluoddity-Core
This is a stripped down version of:
https://github.com/aphid91/Fluoddity
Fluoddity-Core contains just enough machinery to load and run a config (no jitter or parameter sweeps). It is meant as a companion to the full Fluoddity repo for those who want to tinker and/or understand the algorithm without digging through vibe-coded bells and whistles.
For more information see the Readme for Fluoddity

claude coded webgl port demo: https://aphid91.github.io/Fluoddity-Core/
## Structure
System state consists of a particle buffer called "entities" and a texture that stores particle trails called "canvas". 
physics steps work like this:

### Entity Update
- Each particle in entities reads the canvas at a pair of sensor locations.
- The particle extracts the flow/current vector from each sensor reading
- "calculate_entity_behavior()" takes this information and processes it with constants taken from the .json config file (see entity_update.glsl comments for details on this process)
- calculate_entity_behavior outputs a vec2 force and vec2 strafe.
- we update particle state with: velocity =velocity*drag + force; and position += velocity + strafe;
### Brush Update
- In order to write new trails to the canvas, we must splat all the particles to their locations on the canvas.
- A "brush" texture with the same dimensions as canvas acts as a staging area for these newly created trails.
- We use instanced rendering with one instance per entity and additive blending.
- Each particle draws a small gaussian kernel with color == (velocity_x, velocity_y, 0.01, 1) * kernel. (only the velocity terms are used currently, the 0.01 is mostly placeholder)
### Canvas Update
- The canvas update is a simple frag shader. Each frame, the trails diffuse and fade away, while we mix in the newly laid trails from brush.
- diffusion is handled by a simple 4 neighbor weighted average of the canvas
- trail fade is performed by mixing old trails (pre diffused) and new trails (from brush) with canvas_out = trail_persistence*canvas_in + (1-trail_persistence)*brush_in.
- This mix() style trail persistence ensures that the equilibrium trail intensity is independent of the specific trail-persistence value
