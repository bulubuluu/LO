# Non-Commutative Standard Model Amplitude Calculations

This repository contains Python code for calculating scattering amplitudes in the non-commutative extension of the Standard Model. The code implements calculations for both Standard Model (SM) and non-commutative (NC) contributions to 2→2 scattering processes, with a focus on quark-antiquark to Z-boson pair production.

## Purpose

The code provides tools for:
- Calculating SM and NC amplitudes for 2→2 scattering processes
- Calculating the ratios of the amplitudes for two different models, with goal of providing the event reweighting factors and possiblity to build ME-based discriminants to separate different physics models/hypotheses.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/livnczz4l.git
cd livnczz4l
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Basic Usage

The main module `bsm_nc_module.py` provides functions for amplitude calculations:

```python
import bsm_nc_module as nc

# Generate random momenta
p1, p2, k1, k2 = nc.random_momenta()

# Calculate amplitudes
sm_amplitude = nc.sm_amp(p1, k1, k2, index_position="up")
nc_amplitude = nc.nc_amp(p1, k1, k2, index_position="up")
```

See `bsm_nc_example.py` for more detailed examples.

## Code Structure

- `bsm_nc_module.py`: Main module containing amplitude calculations
- `test_bsm_nc_module.py`: Unit tests for the main module
- `bsm_nc_example.py`: Example usage and demonstrations
- `requirements.txt`: Required Python packages

## Contributing

We welcome contributions! Here's how you can help:

1. **Fork the Repository**
   - Click the "Fork" button on GitHub
   - Clone your fork:
     ```bash
     git clone https://github.com/yourusername/livnczz4l.git
     cd livnczz4l
     ```

2. **Set Up Development Environment**
   - Create a new branch for your feature:
     ```bash
     git checkout -b feature/your-feature-name
     ```
   - Install development dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Make Changes**
   - Write your code
   - Add tests in `test_bsm_nc_module.py`
   - Update documentation as needed
   - Follow PEP 8 style guidelines
   - Add type hints to new functions

4. **Test Your Changes**
   ```bash
   pytest test_bsm_nc_module.py
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Description of your changes"
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill in the PR description
   - Submit the PR



## Testing

The code includes a comprehensive test suite. Run tests with:
```bash
pytest test_bsm_nc_module.py
```

Tests cover:
- Momentum conservation
- Gauge invariance
- Physical limits
- Numerical stability
- Input validation

