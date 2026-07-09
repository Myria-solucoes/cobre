//! Interval-overlap primitive on an hour-resolution stage clock.
//!
//! Pure hour arithmetic: no calendar cycle, no dates, no block mode.

/// Intersect `[window_start_hours, window_start_hours + window_width_hours)`
/// against consecutive periods whose durations are `stage_lengths_hours`
/// (period 0 starts at hour `0.0`; each subsequent period starts where the
/// previous one ends).
///
/// Returns one overlap-hours entry per period, from period 0 through the
/// deepest overlapped period — a contiguous `0..=L` range, not a count: a
/// period between two overlapped periods that the window happens to miss
/// keeps its `0.0` entry rather than being dropped. An empty return means the
/// window overlaps no period at all.
///
/// `window_width_hours` and every `stage_lengths_hours` entry must be `>
/// 0.0`, and all inputs must be finite; these are caller preconditions
/// checked with `debug_assert!` in debug builds only.
#[must_use]
pub fn window_period_overlaps(
    window_start_hours: f64,
    window_width_hours: f64,
    stage_lengths_hours: &[f64],
) -> Vec<f64> {
    debug_assert!(
        window_start_hours.is_finite(),
        "window_start_hours must be finite"
    );
    debug_assert!(
        window_width_hours.is_finite() && window_width_hours > 0.0,
        "window_width_hours must be finite and > 0.0"
    );
    debug_assert!(
        stage_lengths_hours
            .iter()
            .all(|&length| length.is_finite() && length > 0.0),
        "every stage_lengths_hours entry must be finite and > 0.0"
    );

    let window_end_hours = window_start_hours + window_width_hours;

    let mut overlaps = Vec::new();
    let mut period_start = 0.0_f64;
    for &length in stage_lengths_hours {
        if period_start >= window_end_hours {
            break;
        }
        let period_end = period_start + length;
        let overlap_start = window_start_hours.max(period_start);
        let overlap_end = window_end_hours.min(period_end);
        overlaps.push((overlap_end - overlap_start).max(0.0));
        period_start = period_end;
    }

    match overlaps.iter().rposition(|&overlap| overlap > 0.0) {
        Some(last_nonzero) => {
            overlaps.truncate(last_nonzero + 1);
            overlaps
        }
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: got {actual:?}, expected {expected:?}"
        );
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() < TOL,
                "value mismatch: got {actual:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn test_uniform_stages_shallow_window() {
        let overlaps = window_period_overlaps(0.0, 360.0, &[168.0, 168.0, 168.0]);
        assert_close(&overlaps, &[168.0, 168.0, 24.0]);
    }

    #[test]
    fn test_non_uniform_stages_depth_three() {
        let overlaps = window_period_overlaps(360.0, 720.0, &[720.0, 168.0, 168.0, 168.0]);
        assert_eq!(overlaps.len(), 4);
        assert!(overlaps[3] > 0.0, "index 3 must be nonzero: {overlaps:?}");
    }

    #[test]
    fn test_skipped_intermediate_stage_stays_contiguous() {
        let overlaps = window_period_overlaps(336.0, 200.0, &[168.0, 168.0, 168.0, 168.0]);
        assert_eq!(overlaps.len(), 4);
        assert_eq!(
            overlaps[1], 0.0,
            "index 1 must be exactly 0.0: {overlaps:?}"
        );
        assert_close(&overlaps, &[0.0, 0.0, 168.0, 32.0]);
    }

    #[test]
    fn test_window_past_all_stages_returns_empty() {
        let overlaps = window_period_overlaps(1000.0, 10.0, &[168.0, 168.0]);
        assert!(overlaps.is_empty(), "expected empty, got {overlaps:?}");
    }

    #[test]
    fn test_exact_boundary_window_returns_depth_zero() {
        let overlaps = window_period_overlaps(0.0, 720.0, &[720.0, 168.0, 168.0]);
        assert_close(&overlaps, &[720.0]);
    }
}
