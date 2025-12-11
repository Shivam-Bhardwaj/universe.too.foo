use crate::coordinates::*;
use crate::grid::*;

#[test]
fn test_cartesian_to_spherical_roundtrip() {
    let positions = [
        CartesianPosition::new(1.0, 0.0, 0.0),
        CartesianPosition::new(0.0, 1.0, 0.0),
        CartesianPosition::new(0.0, 0.0, 1.0),
        CartesianPosition::new(1.0, 1.0, 1.0),
        CartesianPosition::from_au(1.0, 0.0, 0.0),
        CartesianPosition::from_au(5.2, 0.3, -0.1),
    ];

    for pos in positions {
        let spherical = pos.to_spherical();
        let back = spherical.to_cartesian();

        let tolerance = pos.magnitude() * 1e-10; // Relative tolerance
        assert!((pos.x - back.x).abs() < tolerance, "x mismatch");
        assert!((pos.y - back.y).abs() < tolerance, "y mismatch");
        assert!((pos.z - back.z).abs() < tolerance, "z mismatch");
    }
}

#[test]
fn test_shell_boundaries() {
    let grid = HLGGrid::with_defaults();

    // Shell boundaries should be continuous
    for l in 0..50 {
        let outer = grid.shell_outer_radius(l);
        let next_inner = grid.shell_inner_radius(l + 1);
        assert!((outer - next_inner).abs() < 1e-6,
            "Shell {} outer != Shell {} inner", l, l + 1);
    }
}

#[test]
fn test_shell_doubling() {
    let grid = HLGGrid::with_defaults();

    // With base 2, each shell should double in radius
    for l in 0..50 {
        let inner = grid.shell_inner_radius(l);
        let outer = grid.shell_outer_radius(l);
        let ratio = outer / inner;
        assert!((ratio - 2.0).abs() < 1e-10,
            "Shell {} ratio is {} (expected 2.0)", l, ratio);
    }
}

#[test]
fn test_earth_position() {
    let grid = HLGGrid::with_defaults();

    // Earth at 1 AU on x-axis
    let earth = CartesianPosition::from_au(1.0, 0.0, 0.0);
    let cell = grid.cartesian_to_cell(earth).unwrap();

    // Should be in shell ~1 (0.3-0.6 AU is shell 0, 0.6-1.2 AU is shell 1)
    assert!(cell.l <= 2, "Earth should be in shell 0-2, got {}", cell.l);

    // Should be at theta=0 (x-axis), phi=π/2 (equator)
    // theta_idx for theta=0: (0 + π)/(2π) * 64 = 32
    assert!(cell.theta >= 31 && cell.theta <= 33,
        "Earth theta should be ~32, got {}", cell.theta);

    // phi_idx for phi=π/2: (π/2)/π * 32 = 16
    assert!(cell.phi >= 15 && cell.phi <= 17,
        "Earth phi should be ~16, got {}", cell.phi);
}

#[test]
fn test_positions_in_same_cell() {
    let grid = HLGGrid::with_defaults();

    // Two nearby positions should be in the same cell
    let pos1 = CartesianPosition::from_au(1.0, 0.0, 0.0);
    let pos2 = CartesianPosition::from_au(1.001, 0.0, 0.0);

    let cell1 = grid.cartesian_to_cell(pos1).unwrap();
    let cell2 = grid.cartesian_to_cell(pos2).unwrap();

    assert_eq!(cell1, cell2, "Nearby positions should be in same cell");
}

#[test]
fn test_cell_bounds_contain_cell_center() {
    let grid = HLGGrid::with_defaults();

    for l in [0, 5, 10, 20] {
        for theta in [0, 16, 32, 48, 63] {
            for phi in [0, 8, 16, 24, 31] {
                let id = CellId::new(l, theta, phi);
                let bounds = grid.cell_to_bounds(id);

                // Centroid should be inside bounds
                assert!(bounds.centroid.x >= bounds.min.x && bounds.centroid.x <= bounds.max.x);
                assert!(bounds.centroid.y >= bounds.min.y && bounds.centroid.y <= bounds.max.y);
                assert!(bounds.centroid.z >= bounds.min.z && bounds.centroid.z <= bounds.max.z);
            }
        }
    }
}

#[test]
fn test_inside_sun_returns_none() {
    let grid = HLGGrid::with_defaults();

    // Position inside r_min should return None
    let inside = CartesianPosition::new(1e10, 0.0, 0.0); // 10 billion meters, inside Mercury
    assert!(grid.cartesian_to_cell(inside).is_none());
}

#[test]
fn test_far_distances() {
    let grid = HLGGrid::with_defaults();

    // Alpha Centauri at ~4.37 light years
    let alpha_cen = CartesianPosition::new(4.37 * 9.461e15, 0.0, 0.0);
    let cell = grid.cartesian_to_cell(alpha_cen);

    assert!(cell.is_some(), "Should handle interstellar distances");
    let cell = cell.unwrap();

    // At 4.37 ly, should be in a high shell (calculated: ~19-20)
    assert!(cell.l > 15, "Alpha Centauri should be in shell >15, got {}", cell.l);
}
