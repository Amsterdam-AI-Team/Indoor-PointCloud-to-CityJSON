# T3D PointCloud Processing

This repository contains methods for the **automatic detection of rooms in interior PointClouds**. The methods can serve as inspiration, or can be applied as-is.

---

## Project Goal

The goal of this project is to automatically detect rooms in interior point clouds. One of the main challenges in working with interior 3D point cloud data is detecting walls and defining a room boundaries.

This repository contains a five-staged pipeline that preprocesses, split into floors, detects rooms, reconstructs the surface, and computes volume and area

For a quick dive into this repository take a look at our [0. Complete Pipeline](notebooks/0. Complete Pipeline.ipynb).

---

## Folder Structure

 * [`datasets`](./datasets) _Example datasets of interior pointclouds_
 * [`notebooks`](./notebooks) _Jupyter notebook tutorials_
 * [`src`](./src) _Python source code_
   * [`utils`](./src/utils) _Utility functions_
 * [`modules`](./modules) _Pipeline modules_
 * [`preprocessors`](./preprocessors) _Pre-processor modules_

---

## Installation

1. Clone this repository

2. Install all dependencies (requires Python >=3.8):
    ```bash
    pip install -r requirements.txt
    ```

3. Check out the [notebooks](notebooks) for a demonstration.

---

## Acknowledgements

This repository was created by _Falke Boskaljon_ for the City of Amsterdam.