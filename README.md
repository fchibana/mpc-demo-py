# mpc-demo-py

A simple model predictive control (MPC) demo in Python.

In this example the control objective is *point stabilization*, i.e. to drive the robot to a target pose.
<!-- The robot kinematics assume a differential drive robot. -->

The code in this repository is mainly based on the work available [here](https://github.com/MMehrez/MPC-and-MHE-implementation-in-MATLAB-using-Casadi).

## Requirements
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [CasADi](https://web.casadi.org/) 
<!-- - [Python 3.8.x](https://www.python.org/) -->
<!-- - [SciPy](https://scipy.org/) -->

## Basic usage

1. Clone this repository
    ```
    git clone https://github.com/fchibana/mpc-demo-py.git
    ```
2. Install the dependencies
    ```
    pip install -r requirements.txt
    ```

3. Execute `main.py`

### Optional arguments
For details about the available parameters pass the `-h` argument:
```
python3 main.py -h
```

## References
- https://github.com/MMehrez/MPC-and-MHE-implementation-in-MATLAB-using-Casadi
- https://github.com/AtsushiSakai/PythonRobotics

## License
[MIT](https://choosealicense.com/licenses/mit/)