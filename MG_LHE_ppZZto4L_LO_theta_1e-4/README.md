# Non-Commutative Standard Model Amplitude Calculations

This repository contains Python code for calculating scattering amplitudes in the non-commutative extension of the Standard Model. The repository consists of modified and optimized scripts that focus on quark-antiquark to Z-boson pair production processes, enabling targeted event processing, amplitude calculation, and result visualization.

## Purpose

The code provides tools for:
- Processing events with specific quark pairs (1&-1 or 2&-2) and calculating Standard Model (SM) and non-commutative (NC) amplitudes for 2→2 scattering processes
- Filtering events based on predefined criteria and saving only qualified events to output for analysis
- Computing event reweighting factors using the formula Weight = 1 + (NC Amplitude / SM Amplitude)
- Visualizing amplitude and weight distributions via histograms and scatter plots, and identifying extreme events with relevant data export

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LO.git
cd LO
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

Below are the key usage instructions:

1. Event Processing & Amplitude Calculation with `bsm_nc_calculator.py`

This script processes events with specific quark pairs, calculates amplitudes, and saves only filtered events to output.


```python
python bsm_nc_calculator.py <input_file> <tree_name> <output_file>
```

- A LHE-level ROOT file is required to provide the four-momentum information of initial-state and final-state particles
- Processes events containing only specific quark pairs (down quark-antiquark: 1&-1; up quark-antiquark: 2&-2) by filtering via PID
- Saves qualified events to a ROOT file, including key data like momenta, SM/NC amplitudes


2. Visualization & Extreme Event Analysis with `draw-nc-sm-graph.py`

This script reads event data from a ROOT file, computes weights, generates analysis plots, and exports extreme event data to PDF.

```python
python draw-nc-sm-graph.py <input_file> <tree_name>
```

- Calculates and compares amplitudes (SM_Amplitude, NC_Amplitude) for the selected events
- Distributions of weights (1 + NC/SM) to highlight extreme events
- Comparisons of Z boson properties (mass, pT, eta, phi, energy, momentum)
- Scatter plots of SM vs. NC amplitudes
- Exports detailed momentum data (E, Px, Py, Pz) of Z bosons and target quarks from extreme events (weight > 500) to a PDF report 

## Code Structure

- `bsm_nc_calculator.py`: Script for specific quark pair event processing, amplitude calculation, and filtered output
- `draw-nc-sm-graph.py`: Script for data visualization, weight calculation, and extreme event analysis with PDF export
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
     git clone https://github.com/yourusername/LO.git
     cd LO
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

