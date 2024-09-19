# YAMNET

## Setup
### Previous steps for raspberry

1. Install dependencies for HDF5
   ```bash
   sudo apt-get install libhdf5-serial-dev pkg-config
   ```

### On personal PC

1. Install a virtual environment.
   ```bash
   python3 -m venv .venv
   ```

2. Activate the virtual environment
   ```
   source .venv/bin/activate
   ```
3. Install all dependencies.
   ```bash
   pip install -r dependencies.txt
   ```
4. Run model
   ``` bash
   python yamnet.py
   ```
