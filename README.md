## OpenMPC - an open and flexible MPC toolkit for teaching and research

This repo contains OpenMPC, an open, flexible and easy-to-use MPC toolkit for teaching and research. It contains functionality for linear and nonlinear MPC, along with invariant set computations. The optimization relies on cvxpy (for linear MPC) and casadi (for nonlinear MPC). It is compatible with the control package (for both linear and nonlinear systems), and contains invariant set functionality based on cddlib.



## Dependencies

OpenMPC depends on the following Python packages:

* numpy>=1.21.0
* casadi>=3.6.3
* cvxpy[mosek]>=1.3.0
* matplotlib
* control
* pycddlib>=3.0.0b6'


These dependencies will be installed automatically when you run the `pip install . ` command. Some of the provides example use mosek solver which requires a special acesamic license which can be freely installed. Please follow the section on [Mosek License](#mosek-license) carefully.

Once the setup is complete, you can simply run:

```bash
jupyter notebook index.ipynb
```

## Mosek License

You can download your mosek License [here](https://www.mosek.com/resources/getting-started/) by clicking on `Academic License` at line 3. The click on `Request Personal Academic License` and follow the instructios for copying the license file on your personal computer.

DO NOT share your license file with anyone as the license is linked to your name


### Trubleshooting

Since `pycddlib` is in pre-release, you may need to install `cddlib` separately (e.g., using Homebrew):

```bash
brew install cddlib
```

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
