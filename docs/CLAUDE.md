# Fluoddity-Core

WebGL 2 port of Fluoddity. The web app lives in `docs/`.

## UI Focus Gotcha

Browser form controls (`<input type="range">`, `<select>`, `<button>`) retain focus after interaction. A focused control swallows keyboard events before they reach `window.addEventListener('keydown', ...)`, so shortcuts like T, Space, R, G stop working.

**When adding any new interactive UI element**, ensure it loses focus after use:
- Sliders/selects: add a `change` listener that calls `.blur()`
- Buttons: generally fine (canvas mousedown blur handles it), but if in doubt, blur on click
