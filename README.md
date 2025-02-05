## OpenMPC - an open and flexible MPC toolkit for teaching and research

This repo contains OpenMPC, an open, flexible and easy-to-use MPC toolkit for teaching and research. It contains functionality for linear and nonlinear MPC, along with invariant set computations. The optimization relies on cvxpy (for linear MPC) and casadi (for nonlinear MPC). It is compatible with the control package (for both linear and nonlinear systems), and contains invariant set functionality based on cddlib.



## Dependencies

OpenMPC depends on the following Python packages:

* numpy>=1.21.0
* casadi>=3.6.3
* cvxpy>=1.3.0
* matplotlib
* control
* pycddlib>=3.0.0b6'


These dependencies will be installed automatically when you run the `pip install . ` command.
It is possible that your system does not have all the requirements for installying pycddlib. Indeed, pycddlib depends on cddlib and GMP which you should install as propsed [here](https://github.com/mcmtroffaes/pycddlib/blob/develop/INSTALL.rst)


On Fedora
```
dnf install cddlib-devel gmp-devel python3-devel
```

On Ubuntu

```
apt-get install libcdd-dev libgmp-dev python3-dev
```

On Mac

```
brew install cddlib gmp
```

For Windows there is not need to for further installations.


Once the setup is complete, you can simply run:

```bash
jupyter notebook index.ipynb
```


**Note.** Since `pycddlib` is in pre-release, you may need to install `cddlib` separately (e.g., using Homebrew):

```bash
brew install cddlib
```


The latest version of `pycddlib` is then built from source. If you need to specify the include paths, you can do this using:

```bash
export CFLAGS="-I/usr/local/include"
export LDFLAGS="-L/usr/local/lib"
pip install .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

If you wish to contribute to this project, please feel free to open a pull request or an issue on the repository.
