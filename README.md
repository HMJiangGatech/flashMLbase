# flashMLbase
A lightweight reusable C++ Machine Learning Code Base. It is built upon Eigen, which provide efficient matrix computation operator.

## Model Zoo
**PISTA**: Pathwise Iterative Shrinkage Thresholding Algorithm for Lasso

**PIS$^2$TA**: PISTA for SQRT Lasso

**CMR**: Calibrated Multivariate Regression

## Usage
An example of synthetic data is in `Tasks\Synthetic\`

- First, you need to write a file containing model configuration.
- Then you may rewrite `runCLITask` in `src\cli_main.cpp`.
- Complie your program via `CMake`.
- Run `fmlbase` or `fmlbase.exe`
