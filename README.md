# pr3

**pr**actical **pr**ojection **p**ursuit **r**egression

## Setup

On macOS:

* [Install Homebrew](https://brew.sh).
* Install Python with `brew install python@3.8`; if you have conflicting versions, `brew unlink` and `brew link` as needed.
* Run `make`.


## About

This repository offers an implementation of
[projection pursuit regression](https://en.wikipedia.org/wiki/Projection_pursuit_regression) with a handful of tweaks
for ease-of-use such as inspectability of the model through some convenient plotting functionality. For a more
established implementation we point to [`projection-pursuit`](https://github.com/pavel-aicradle/projection-pursuit),
which includes a more fully-fledged feature suite (e.g., multiple output dimensions, backfitting). Some interesting
components of this repository's implementation include:

* in the linear step of the alternating minimization (to compute an optimal projection direction), we introduce the
 capability to optimize for _sparse_ projections, which helps for understanding the role of each projection; that is,
 a linear combination of, say, three coordinates is easier to understand than a linear combination of hundreds;
* in the nonlinear step of the alternating minimization (to compute an optimal ridge function), we supplement the
 standard capability for polynomial regression with options for piecewise linear regression (through a one-hidden-layer
 perceptron with rectilinear activation) and kernel regression (Nadaraya-Watson).
