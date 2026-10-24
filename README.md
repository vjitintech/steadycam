# SteadyCam v2

Live, in-camera video stabilization for phones, done entirely in software with no
gimbal, tripod or arm. Frames are stabilized as they are captured, so there is no
post-processing step.

- **Smooth**: removes shake and keeps intentional pans.
- **Tripod**: holds the framing still, including subject size when the phone moves forward or back.
- **Target lock**: tap a subject and it stays fixed on screen while the phone moves.
  Camera-motion prediction plus template confirmation, with LOST/re-acquire handling.
- 4K capture with clean 1080p output (or 720p / 4K), 30/60 fps, lens + digital zoom
  (slider, pinch), flip camera, torch, tap-to-focus.
- Records the stabilized live view and auto-saves when recording stops.

## Files
- `index.html`: the app. Open over HTTPS in Safari or Chrome and tap **Camera**.
- `steadycam.js`: plugin SDK for any web camera app (`SteadyCam.install()`).
- `src/`: engine and SDK source. `test/`: `node test/test.js` and `node test/sdk-test.js`.

Proprietary. All rights reserved. See LICENSE.
