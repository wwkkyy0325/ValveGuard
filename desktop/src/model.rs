//! Portable JSON tree ensemble exported by `export_model_json.py`.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
pub struct Tree {
    feature: Vec<i32>,
    threshold: Vec<f64>,
    left: Vec<i32>,
    right: Vec<i32>,
    leaf: Vec<bool>,
    missing_left: Vec<bool>,
    value: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Model {
    pub version: String,
    pub features: Vec<String>,
    pub threshold: f64,
    pub prior: f64,
    pub k0: f64,
    baseline: f64,
    trees: Vec<Tree>,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read model {}", path.display()))?;
        serde_json::from_str(&text).context("parse model json")
    }

    pub fn score(&self, x: &[f64]) -> f64 {
        let mut s = self.baseline;
        for t in &self.trees {
            let mut i = 0usize;
            while !t.leaf[i] {
                let f = t.feature[i] as usize;
                let go_left = match x.get(f) {
                    Some(v) if v.is_finite() => *v <= t.threshold[i],
                    _ => t.missing_left[i],
                };
                i = if go_left { t.left[i] } else { t.right[i] } as usize;
            }
            s += t.value[i];
        }
        1.0 / (1.0 + (-s).exp())
    }
}
