## OpenMPC - an open and flexible MPC toolkit for teaching and research

This repo contains OpenMPC, an open, flexible and easy-to-use MPC toolkit for teaching and research. It contains functionality for linear and nonlinear MPC, along with invariant set computations. The optimization relies on cvxpy (for linear MPC) and casadi (for nonlinear MPC). It is compatible with the control package (for both linear and nonlinear systems), and contains invariant set functionality based on cddlib.

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

Once the setup is complete, you can simply run:

```bash
jupyter notebook index.ipynb
```

## Dependencies

OpenMPC depends on the following Python packages:

-numpy>=1.21.0
-casadi>=3.6.3
-cvxpy>=1.3.0
-matplotlib
-control


These dependencies will be installed automatically when you run the `pip install . ` command.

**Note.** If you encounter any issues during installation, refer to the `pycddlib` documentation for additional guidance on building and installing pycddlib.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

If you wish to contribute to this project, please feel free to open a pull request or an issue on the repository.
