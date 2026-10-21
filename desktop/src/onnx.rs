//! ONNX Runtime wrapper for the exported CNN ensemble.

use anyhow::{Context, Result};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

use crate::channels::CHANNELS;
use crate::features::LEN;

pub struct Ensemble {
    _environment: Environment,
    session: Session,
}

impl Ensemble {
    pub fn load(path: &Path) -> Result<Self> {
        let environment = ort::init().with_name("valveguard").build()?;
        let session = Session::builder(&environment)?
            .commit_from_file(path)
            .with_context(|| format!("load ONNX model {}", path.display()))?;
        Ok(Self {
            _environment: environment,
            session,
        })
    }

    pub fn score(&mut self, windows: &[Vec<[f32; CHANNELS]>]) -> Result<Vec<f32>> {
        let count = windows.len();
        let mut data = Vec::with_capacity(count * LEN * CHANNELS);
        for window in windows {
            if window.len() != LEN {
                anyhow::bail!("expected {LEN} ticks, got {}", window.len());
            }
            for tick in window {
                data.extend_from_slice(&tick[..]);
            }
        }
        let tensor = Tensor::from_array(([count, LEN, CHANNELS], data.into_boxed_slice()))?;
        let outputs = self.session.run(ort::inputs!["channels" => tensor])?;
        let (_, scores) = outputs["scores"].try_extract_tensor::<f32>()?;
        Ok(scores.to_vec())
    }
}
