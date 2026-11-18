//! Port of `sympy_extras/settings.py`: the global settings of the
//! package (one per thread here) and [`configure`], which changes them
//! until its guard is dropped. Unlike the Python, `configure` checks its
//! arguments before it changes anything: the Python changes the settings
//! given before an invalid `groebner_algorithm` or `time_scale` and does
//! not restore them.

use std::cell::RefCell;

use crate::error::{value_error, PolyResult};

/// The algorithm of the direct computation of a Gröbner basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroebnerAlgorithm {
    F4,
    Buchberger,
}

impl GroebnerAlgorithm {
    /// The algorithm of its name (`'f4'` or `'buchberger'`).
    pub fn named(name: &str) -> PolyResult<GroebnerAlgorithm> {
        match name {
            "f4" => Ok(GroebnerAlgorithm::F4),
            "buchberger" => Ok(GroebnerAlgorithm::Buchberger),
            _ => Err(value_error(format!(
                "groebner_algorithm is 'f4' or 'buchberger', got '{}'",
                name
            ))),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            GroebnerAlgorithm::F4 => "f4",
            GroebnerAlgorithm::Buchberger => "buchberger",
        }
    }
}

/// The settings (`Settings`).
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Whether certified numerical evaluation may decide signs.
    pub numerical_checks: bool,
    /// The significant digits of the numerical checks.
    pub precision: usize,
    /// The default time limit in seconds, `None` for no limit.
    pub timeout: Option<f64>,
    /// The factor of every time limit.
    pub time_scale: f64,
    /// Whether Gröbner bases over the rationals may go modular.
    pub modular_groebner: bool,
    /// The seconds of the direct computation before the modular one.
    pub groebner_direct_time: f64,
    /// The algorithm of the direct computation.
    pub groebner_algorithm: GroebnerAlgorithm,
}

/// The value of `SYMPY_EXTRAS_TIME_SCALE`: `1` when it is not set or not
/// a positive number (`_time_scale_of_the_environment`).
pub fn time_scale_of_the_environment() -> f64 {
    let given: String = std::env::var("SYMPY_EXTRAS_TIME_SCALE").unwrap_or_default();
    match given.trim().parse::<f64>() {
        Ok(scale) if scale > 0.0 && scale.is_finite() => scale,
        _ => 1.0,
    }
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            numerical_checks: true,
            precision: 30,
            timeout: Some(30.0),
            time_scale: time_scale_of_the_environment(),
            modular_groebner: true,
            groebner_direct_time: 0.25,
            groebner_algorithm: GroebnerAlgorithm::F4,
        }
    }
}

thread_local! {
    static CURRENT: RefCell<Settings> = RefCell::new(Settings::default());
}

/// The current settings (a copy).
pub fn settings() -> Settings {
    CURRENT.with(|s| s.borrow().clone())
}

/// The changes of [`configure`]: `None` leaves a setting unchanged;
/// `no_timeout` removes the time limit.
#[derive(Clone, Debug, Default)]
pub struct Changes {
    pub numerical_checks: Option<bool>,
    pub precision: Option<usize>,
    pub timeout: Option<f64>,
    pub no_timeout: bool,
    pub modular_groebner: Option<bool>,
    pub groebner_direct_time: Option<f64>,
    pub groebner_algorithm: Option<String>,
    pub time_scale: Option<f64>,
}

/// The guard of [`configure`]: the previous settings come back when it is
/// dropped.
pub struct Configured {
    previous: Settings,
}

impl Drop for Configured {
    fn drop(&mut self) {
        let previous: Settings = self.previous.clone();
        CURRENT.with(|s| *s.borrow_mut() = previous);
    }
}

/// The settings changed until the guard is dropped (`configure`).
pub fn configure(changes: &Changes) -> PolyResult<Configured> {
    if let Some(scale) = changes.time_scale {
        if scale.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(value_error("time_scale must be positive".to_string()));
        }
    }
    let algorithm: Option<GroebnerAlgorithm> = changes
        .groebner_algorithm
        .as_deref()
        .map(GroebnerAlgorithm::named)
        .transpose()?;
    let previous: Settings = settings();
    CURRENT.with(|s| {
        let mut s = s.borrow_mut();
        if let Some(scale) = changes.time_scale {
            s.time_scale = scale;
        }
        if let Some(v) = changes.numerical_checks {
            s.numerical_checks = v;
        }
        if let Some(v) = changes.precision {
            s.precision = v;
        }
        if let Some(v) = changes.timeout {
            s.timeout = Some(v);
        }
        if changes.no_timeout {
            s.timeout = None;
        }
        if let Some(v) = changes.modular_groebner {
            s.modular_groebner = v;
        }
        if let Some(v) = changes.groebner_direct_time {
            s.groebner_direct_time = v;
        }
        if let Some(a) = algorithm {
            s.groebner_algorithm = a;
        }
    });
    Ok(Configured { previous })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configure_restores() {
        let before: Settings = settings();
        {
            let _g = configure(&Changes {
                numerical_checks: Some(false),
                no_timeout: true,
                ..Changes::default()
            })
            .unwrap();
            assert!(!settings().numerical_checks);
            assert_eq!(settings().timeout, None);
        }
        assert_eq!(settings(), before);
        let bad = Changes {
            numerical_checks: Some(false),
            groebner_algorithm: Some("x".to_string()),
            ..Changes::default()
        };
        assert!(configure(&bad).is_err());
        assert_eq!(settings(), before);
        assert!(configure(&Changes {
            time_scale: Some(0.0),
            ..Changes::default()
        })
        .is_err());
    }
}
