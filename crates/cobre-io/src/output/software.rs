//! The identity this build stamps into every output it writes.

/// Recorded as `software` in output metadata and policy checkpoints.
pub const SOFTWARE_NAME: &str = "cobre-myria";

/// Fork release identity; deliberately distinct from the upstream ABI version.
pub const SOFTWARE_VERSION: &str = "0.18.0-myria.1";

/// How to obtain a policy checkpoint this build loads; every refusal that
/// says how ends with it.
#[must_use]
pub fn policy_checkpoint_remedy() -> String {
    format!(
        "re-run the program that produced it with {SOFTWARE_NAME} {SOFTWARE_VERSION}; \
         for a converted boundary policy, convert it again"
    )
}

/// The software an output records as its writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftwareIdentity<'a> {
    /// `None` when the output predates recording the name.
    pub name: Option<&'a str>,
    /// The writer's version string, compared verbatim.
    pub version: &'a str,
}

impl SoftwareIdentity<'static> {
    /// This build.
    pub const THIS_BUILD: Self = Self {
        name: Some(SOFTWARE_NAME),
        version: SOFTWARE_VERSION,
    };
}
