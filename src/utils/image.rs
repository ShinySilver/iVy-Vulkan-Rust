use fastnoise2::SafeNode;
use glam::{uvec2, UVec2};
use image::{GrayImage, Luma};
use log::warn;

#[derive(Default, Clone)]
pub struct Img<T> {
    pub width: u32,
    pub data: Vec<T>,
}

impl<T: Clone> Img<T> {
    pub fn new(width: u32, value: T) -> Self {
        Self {
            width,
            data: vec![value; (width * width) as usize],
        }
    }
}

impl<T> Img<T> {
    pub fn from_fn<F>(width: u32, mut f: F) -> Self
    where
        F: FnMut(u32, u32) -> T,
    {
        let mut data = Vec::with_capacity((width * width) as usize);
        for y in 0..width {
            for x in 0..width {
                data.push(f(x, y));
            }
        }
        Self { width, data }
    }

    pub fn get(&self, pos: UVec2) -> &T {
        &self.data[(pos.x + pos.y * self.width) as usize]
    }

    pub fn get_mut(&mut self, pos: UVec2) -> &mut T {
        &mut self.data[(pos.x + pos.y * self.width) as usize]
    }

    pub fn set(&mut self, pos: UVec2, value: T) {
        self.data[(pos.x + pos.y * self.width) as usize] = value;
    }

    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(u32, u32, &T),
    {
        for (i, val) in self.data.iter().enumerate() {
            let x = (i as u32) % self.width;
            let y = (i as u32) / self.width;
            f(x, y, val);
        }
    }

    pub fn map<F, T2>(&self, mut f: F) -> Img<T2>
    where
        F: FnMut(u32, u32, &T) -> T2,
    {
        let mut data = Vec::with_capacity(self.data.len());
        for (i, v) in self.data.iter().enumerate() {
            let x = (i as u32) % self.width;
            let y = (i as u32) / self.width;
            data.push(f(x, y, v));
        }
        Img {
            data,
            width: self.width,
        }
    }
}

impl Img<f32> {
    pub fn from_node(node: &SafeNode, width: u32, scaling: f32, seed: i32) -> Self {
        let mut data = vec![0.0f32; (width * width) as usize];
        node.gen_uniform_grid_2d(
            &mut data,
            (width as i32) / -2,
            (width as i32) / -2,
            width as i32,
            width as i32,
            scaling * (0.8e-2),
            seed,
        );
        Self {
            width,
            data,
        }
    }
}

impl Img<u8> {
    pub fn from_file(path: &str) -> Self { Self { data: Default::default(), width: 0 } }

    pub fn to_file(&self, path: &str) {
        GrayImage::from_fn(self.width, self.width, |x, y| {
            Luma([*self.get(uvec2(x, y))])
        }).save(path).unwrap_or_else(|_| warn!("Failed to save {}", path));
    }

    pub fn to_file_and_show(&self, path: &str) {
        self.to_file(path);
        opener::open(path).expect("Failed to open file");
    }
}