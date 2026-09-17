# Evaluation summary

All geometry values were measured on the same 2,702 frames from seven
hairstyle sequences. Point penetration uses winding-number signed distance on
valid non-root points. Segment collision uses exact Embree segment-triangle
intersection and excludes the root-adjacent segment.

| Metric | Init | QP baseline | HairADMM one-step |
| --- | ---: | ---: | ---: |
| Point penetrations | 1,275 | 442 | **15** |
| Segment intersections | 97,490 | 31,598 | **127** |
| Per-frame runtime | — | **~300 s** (historical estimate) | **4.119 s** (instrumented algorithm time) |

Relative to QP, HairADMM reduces point penetrations by 96.61%, segment
intersections by 99.60%, and has a representative speed ratio of approximately
72.8x.

The runtime comparison is intentionally qualified. The QP value is a
historical wall-clock engineering estimate; no complete solver-internal QP
timing series was retained. QP output timestamps provide a partial
cross-check: median adjacent-output intervals are 282.5 s for `curly` and
318.7 s for `girl_long`, while simpler sequences are faster. HairADMM's
4.119 s/frame is the solver-recorded algorithm mean and excludes unrelated
I/O and offline evaluation.

The method is not a strict global non-collision guarantee. All 127 residual
segment intersections are concentrated in the `girl_long` sequence.
