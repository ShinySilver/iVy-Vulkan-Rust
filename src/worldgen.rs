use half::f16;
use crate::world::{World, WorldGenerator};

struct ImageU8 {
    width: u32,
    data: Vec<u8>,
}

impl ImageU8 {
    #[cfg(feature = "debug-view")]
    fn display(&self, title: &str) {
        let window = show_image::create_window(title, Default::default()).unwrap();
        let image = show_image::ImageView::new(show_image::ImageInfo::mono8(self.width, self.width), &self.data);
        window.set_image("grayscale", image).unwrap();
    }
}

struct ImageF16 {
    width: u32,
    data: Vec<f16>,
}

impl ImageF16 {
    fn to_u8(&self, min: f32, max: f32) -> ImageU8 {
        let scale = 255.0 / (max - min);
        let quantized = self.data.iter()
            .map(|v| ((v - min) * scale).clamp(0.0, 255.0) as u8)
            .collect();
        ImageU8 {
            width: self.width,
            data: quantized,
        }
    }

    fn to_gradient(&self, min: f32, max: f32) -> ImageU8 {
        let width = self.width as usize;
        let height = self.data.len() / width;
        let mut grad = vec![0.0f32; self.data.len()];
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let i = y * width + x;
                let dx = (self.data[i + 1] - self.data[i - 1]) * 0.5;
                let dy = (self.data[i + width] - self.data[i - width]) * 0.5;
                grad[i] = (dx * dx + dy * dy).sqrt();
            }
        }
        ImageF16 {
            width: self.width,
            data: grad.iter().map(|&v| f16::from_f32(v)).collect(),
        }
    }
}

impl WorldGenerator for World{

}