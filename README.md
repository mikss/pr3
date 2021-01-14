# pr3

**pr**actical **pr**ojection **p**ursuit **r**egression

## Setup

* [Install Homebrew](https://brew.sh).
* Install Python with `brew install python@3.8`; if you have conflicting versions, `brew unlink` and `brew link` as needed.
* Run `make`.


## About

This repository offers an implementation of
[projection pursuit regression](https://en.wikipedia.org/wiki/Projection_pursuit_regression) with a handful of tweaks
for ease-of-use such as inspectability of the model through some convenient plotting functionality. For a more
established implementation we point to [`projection-pursuit`](https://github.com/pavel-aicradle/projection-pursuit),
which includes a more fully-fledged feature suite (e.g., multiple output dimensions, backfitting).
