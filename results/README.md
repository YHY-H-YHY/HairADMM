# Evaluation summary

All values were measured on the same 2,702 frames from seven hairstyle
sequences. Point penetration uses winding-number signed distance on valid
non-root points. Segment collision uses exact Embree segment-triangle
intersection and excludes the root-adjacent edge. Temporal acceleration is
`||x[t+1] - 2 x[t] + x[t-1]||` in millimetres.

| Metric | Init | QP baseline | HairADMM one-step |
| --- | ---: | ---: | ---: |
| Point penetrations | 1,275 | 442 | **15** |
| Segment intersections | 97,490 | 31,598 | **127** |
| Mean temporal acceleration | 0.635 mm | 1.125 mm | **0.603 mm** |
| Mean algorithm time | — | not retained | **4.119 s/frame** |

Relative to QP, HairADMM reduces point penetrations by 96.61%, segment
intersections by 99.60%, and mean temporal acceleration by 46.40%.

The method is not a strict global non-collision guarantee. The measured
one-step release has a 72.72 mm worst-case temporal outlier in `curly` frame
57, and all 127 residual segment intersections are concentrated in the
`girl_long` sequence. These limitations are reported rather than hidden.
