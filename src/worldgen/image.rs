use glam::{uvec2, UVec2, UVec3};
use half::f16;
use image::{GrayImage, Luma};
use log::warn;

pub struct ImageU8 {
    pub width: u32,
    pub data: Vec<u8>,
}

impl ImageU8 {
    pub fn get(&self, pos: UVec2) -> u8 { self.data[(pos.x + pos.y * self.width) as usize] }
    pub fn set(&mut self, pos: UVec2, value: u8) { self.data[(pos.x + pos.y * self.width) as usize] = value; }
    pub fn export(&self, title: &str) {
        GrayImage::from_fn(self.width, self.width, |x, y| {
            Luma([self.get(uvec2(x, y))])
        }).save(title).unwrap_or_else(|_| warn!("Failed to save {}", title));
        //opener::open(title).unwrap_or_else(|_| warn!("Failed to open {}", title));
    }
}

pub struct ImageF16 {
    pub width: u32,
    pub data: Vec<f16>,
}

impl ImageF16 {
    pub fn get(&self, pos: &UVec2) -> f16 { self.data[(pos.x + pos.y * self.width) as usize] }
    pub fn set(&mut self, pos: &UVec2, value: f16) { self.data[(pos.x + pos.y * self.width) as usize] = value; }
    pub fn to_u8(&self, min: f32, max: f32) -> ImageU8 {
        let scale = 255.0 / (max - min);
        let quantized = self.data.iter()
            .map(|v| ((v.to_f32() - min) * scale).clamp(0.0, 255.0) as u8)
            .collect();
        ImageU8 {
            width: self.width,
            data: quantized,
        }
    }
    pub fn to_gradient(&self, min: f32, max: f32) -> ImageF16 {
        let width = self.width as usize;
        let height = self.data.len() / width;
        let mut grad = vec![0.0f32; self.data.len()];
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let i = y * width + x;
                let dx = (self.data[i + 1] - self.data[i - 1]).to_f32()* 0.5;
                let dy = (self.data[i + width] - self.data[i - width]).to_f32() * 0.5;
                grad[i] = (dx * dx + dy * dy).sqrt();
            }
        }
        ImageF16 {
            width: self.width,
            data: grad.iter().map(|&v| f16::from_f32(v)).collect(),
        }
    }
}