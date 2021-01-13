"""TODO
* Pursuit design:
    - See use of mixins from
    [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html) and also
    [skpp](https://github.com/pavel-aicradle/projection-pursuit/blob/master/skpp/skpp.py).
    - Design the alternating minimization scheme, with all the appropriate stopping criteria as-needed.
    - Introduce plotting functionality to the class.
    - Allow pass thru of nonlinear_kwargs to nonlinear.
* Benchmark examples:
    - visualize w/ better plots;
    - warn about _post hoc_ "just so" stories;
    - label components by iteration step;
    - link to [wiki](https://en.wikipedia.org/wiki/Projection_pursuit_regression);
    - discuss pros/cons of the various choices you can make.
"""
