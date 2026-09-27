use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum SamplerError {
    #[error("logits must be non-empty and finite")]
    InvalidLogits,
    #[error("temperature must be finite and non-negative")]
    InvalidTemperature,
    #[error("top-k must be positive")]
    InvalidTopK,
    #[error("top-p must be in (0, 1]")]
    InvalidTopP,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
        }
    }
}

pub struct Sampler {
    config: SamplerConfig,
    state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig, seed: u64) -> Result<Self, SamplerError> {
        validate_config(config)?;
        Ok(Self {
            config,
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        })
    }

    pub fn config(&self) -> SamplerConfig {
        self.config
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<u32, SamplerError> {
        if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
            return Err(SamplerError::InvalidLogits);
        }
        if self.config.temperature == 0.0 {
            return Ok(argmax(logits));
        }
        let mut order = (0..logits.len()).collect::<Vec<_>>();
        order.sort_by(|left, right| {
            logits[*right]
                .partial_cmp(&logits[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        order.truncate(self.config.top_k.min(order.len()));
        let maximum = logits[order[0]];
        let mut weights = order
            .iter()
            .map(|index| ((logits[*index] - maximum) / self.config.temperature).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f32>();
        if !total.is_finite() || total <= 0.0 {
            return Err(SamplerError::InvalidLogits);
        }
        let mut cumulative = 0.0;
        let mut selected = 0;
        for (index, weight) in weights.iter().enumerate() {
            cumulative += *weight / total;
            selected = index;
            if cumulative >= self.config.top_p {
                break;
            }
        }
        weights.truncate(selected + 1);
        let selected_total = weights.iter().sum::<f32>();
        let target = self.next_f64() as f32 * selected_total;
        let mut running = 0.0;
        for (index, weight) in weights.iter().enumerate() {
            running += *weight;
            if target <= running {
                return Ok(order[index] as u32);
            }
        }
        Ok(order[selected] as u32)
    }

    fn next_f64(&mut self) -> f64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        let bits = value.wrapping_mul(0x2545_f491_4f6c_dd1d);
        ((bits >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

fn validate_config(config: SamplerConfig) -> Result<(), SamplerError> {
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err(SamplerError::InvalidTemperature);
    }
    if config.top_k == 0 {
        return Err(SamplerError::InvalidTopK);
    }
    if !config.top_p.is_finite() || config.top_p <= 0.0 || config.top_p > 1.0 {
        return Err(SamplerError::InvalidTopP);
    }
    Ok(())
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    best as u32
}
