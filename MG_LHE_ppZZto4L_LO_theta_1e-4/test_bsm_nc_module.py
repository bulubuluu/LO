"""
Unit tests for bsm_nc_module.py.

This module contains tests for the non-commutative Standard Model amplitude
calculations, including tests for:
- Momentum conservation
- Gauge invariance
- Physical limits
- Numerical stability
- Input validation
"""

import pytest
import numpy as np
import bsm_nc_module as nc
from bsm_nc_module import (
    PHYSICS, MODEL, FourVector,
    random_momenta, mand_s, mand_t, mand_u,
    sm_amp, nc_amp, term01, term11, term12, term13, term14,
    term21, term22, term23, term24, term25, term26, theta_avg, theta_dual,
    index_switch, avg, contract, eps
)

# Test fixtures
@pytest.fixture
def sample_momenta():
    """Generate a set of test momenta that satisfy energy-momentum conservation."""
    return random_momenta()

@pytest.fixture
def zero_momenta():
    """Generate a set of zero momenta for testing edge cases."""
    return [np.zeros(4) for _ in range(4)]

def sample_onshell_momenta():
    return nc.random_momenta_onshell()

def test_momentum_conservation(sample_momenta):
    """Test that the generated momenta satisfy energy-momentum conservation."""
    p1, p2, k1, k2 = sample_momenta
    total_in = p1 + p2
    total_out = k1 + k2
    
    # Check that total incoming equals total outgoing momentum
    assert np.allclose(total_in, total_out, rtol=1e-10, atol=1e-10), \
        "Momentum conservation violated"

def test_mandelstam_relations(sample_momenta):
    """Test that Mandelstam variables satisfy s + t + u = 2*M_Z^2."""
    p1, p2, k1, k2 = sample_onshell_momenta()
    s = mand_s(p1, p2, k1, k2)
    t = mand_t(p1, p2, k1, k2)
    u = mand_u(p1, p2, k1, k2)
    
    expected_sum = 2 * PHYSICS.M_Z**2
    actual_sum = s + t + u
    
    assert np.isclose(actual_sum, expected_sum, rtol=1e-10, atol=1e-10), \
        f"Mandelstam relation violated: s + t + u = {actual_sum}, expected {expected_sum}"

def test_sm_amp_gauge_invariance(sample_momenta):
    """Test that SM amplitude is gauge invariant (unchanged under p1 -> p1 + k)."""
    p1, p2, k1, k2 = sample_onshell_momenta()
    original_amp = sm_amp(p1, k1, k2, "up")
    
    # Add a multiple of k1 to p1 (gauge transformation)
    p1_transformed = p1 + 0.1 * k1
    transformed_amp = sm_amp(p1_transformed, k1, k2, "up")
    
    assert np.isclose(original_amp, transformed_amp, rtol=1e-10, atol=1e-10), \
        f"SM amplitude not gauge invariant. SM amplitude: {original_amp}, transformed amplitude: {transformed_amp}"

def test_nc_amp_theta_limit():
    """Test that NC amplitude reduces to SM amplitude when theta -> 0."""
    p1, p2, k1, k2 = sample_onshell_momenta()
    
    # Store original theta
    original_theta = MODEL.THETA.copy()
    
    # Set theta to zero
    MODEL.THETA = np.zeros((4, 4))
    nc_amp_zero_theta = nc_amp(p1, k1, k2, "up")
    
    # Restore original theta
    MODEL.THETA = original_theta
    
    # Calculate SM amplitude
    sm_amp_value = sm_amp(p1, k1, k2, "up")
    
    assert np.isclose(nc_amp_zero_theta + sm_amp_value, sm_amp_value, rtol=1e-10, atol=1e-10), \
        "NC amplitude does not reduce to SM amplitude in theta -> 0 limit"

def test_index_position_validation():
    """Test that amplitude functions validate index_position parameter."""
    p1, p2, k1, k2 = sample_onshell_momenta()
    
    # Test valid values
    sm_amp(p1, k1, k2, "up")
    sm_amp(p1, k1, k2, "down")
    nc_amp(p1, k1, k2, "up")
    nc_amp(p1, k1, k2, "down")
    
    # Test invalid values
    with pytest.raises(ValueError):
        sm_amp(p1, k1, k2, "invalid")
    with pytest.raises(ValueError):
        nc_amp(p1, k1, k2, "invalid")

def test_momentum_index_switching(sample_momenta):
    """Test that index switching works correctly for momenta."""
    p1, _, _, _ = sample_momenta
    
    # Switch indices twice should return original vector
    p1_switched_twice = index_switch(index_switch(p1))
    assert np.allclose(p1, p1_switched_twice, rtol=1e-10, atol=1e-10), \
        "Double index switching should return original vector"

def test_theta_dual_properties(sample_momenta):
    """Test properties of theta-dual operation."""
    p1, _, _, _ = sample_momenta
    
    # Test linearity
    p2 = np.random.rand(4)
    alpha, beta = 0.5, 0.7
    
    dual_sum = theta_dual(alpha * p1 + beta * p2)
    sum_of_duals = alpha * theta_dual(p1) + beta * theta_dual(p2)
    
    assert np.allclose(dual_sum, sum_of_duals, rtol=1e-10, atol=1e-10), \
        "theta_dual operation is not linear"

def test_avg_antisymmetry(sample_momenta):
    """Test antisymmetry property of the avg function."""
    p1, p2, k1, k2 = sample_momenta
    
    # avg should be antisymmetric under exchange of any two arguments
    original = float(avg(p1, p2, k1, k2))
    swapped = float(avg(p2, p1, k1, k2))
    
    assert np.isclose(original, -swapped, rtol=1e-10, atol=1e-10), \
        "avg function is not antisymmetric"

def test_numerical_stability():
    """Test numerical stability with extreme momentum values."""
    # Test with very large momenta
    large_momenta = [np.array([1e10, 1e10, 1e10, 1e10]) for _ in range(4)]
    p1, p2, k1, k2 = large_momenta
    
    # Should not raise any numerical errors
    sm_amp(p1, k1, k2, "up")
    nc_amp(p1, k1, k2, "up")
    
    # Test with very small momenta
    small_momenta = [np.array([1e-10, 1e-10, 1e-10, 1e-10]) for _ in range(4)]
    p1, p2, k1, k2 = small_momenta
    
    # Should not raise any numerical errors
    sm_amp(p1, k1, k2, "up")
    nc_amp(p1, k1, k2, "up")

def test_individual_terms_consistency(sample_momenta):
    """Test consistency between individual terms and total NC amplitude."""
    p1, p2, k1, k2 = sample_momenta
    s = mand_s(p1, p2, k1, k2)
    t = mand_t(p1, p2, k1, k2)
    u = mand_u(p1, p2, k1, k2)
    
    # Calculate sum of individual terms
    terms_sum = (
        float(term01(p1, p2, k1, k2, s, t, u)) +
        float(term11(p1, p2, k1, k2, s, t, u)) +
        float(term12(p1, p2, k1, k2, s, t, u)) +
        float(term13(p1, p2, k1, k2, s, t, u)) +
        float(term14(p1, p2, k1, k2, s, t, u)) +
        float(term21(p1, p2, k1, k2, s, t, u)) +
        float(term22(p1, p2, k1, k2, s, t, u)) +
        float(term23(p1, p2, k1, k2, s, t, u)) +
        float(term24(p1, p2, k1, k2, s, t, u)) +
        float(term25(p1, p2, k1, k2, s, t, u)) +
        float(term26(p1, p2, k1, k2, s, t, u))
    )
    
    # Calculate total NC amplitude
    total_amp = float(nc_amp(p1, k1, k2, "up") * 6.0)  # Multiply by normalization factor
    
    # Compare first three terms with total amplitude
    # Note: This is a partial test as we're only checking first three terms
    assert np.isclose(terms_sum, total_amp, rtol=1e-10, atol=1e-10), \
        "Sum of individual terms does not match total amplitude"

if __name__ == "__main__":
    pytest.main([__file__]) 