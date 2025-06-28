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

impl Img<u8> {
    pub fn fast_distance_transform(&self) -> Img<f32> {
        let mut result = self.map(|_, _, &v| { if v == 0 { 0.0 } else { f32::INFINITY } });

        let width = self.width as i32;
        let height = self.data.len() as i32 / width;

        let get = |img: &Img<f32>, x: i32, y: i32| -> f32 {
            if x >= 0 && y >= 0 && x < width && y < height {
                *img.get(uvec2(x as u32, y as u32))
            } else {
                f32::INFINITY
            }
        };

        // Forward pass
        for y in 0..height {
            for x in 0..width {
                let p = uvec2(x as u32, y as u32);
                let mut d = *result.get(p);
                if d > 0.0 {
                    d = d.min(get(&result, x - 1, y) + 1.0);
                    d = d.min(get(&result, x, y - 1) + 1.0);
                    d = d.min(get(&result, x - 1, y - 1) + (2.0f32).sqrt());
                    d = d.min(get(&result, x + 1, y - 1) + (2.0f32).sqrt());
                    result.set(p, d);
                }
            }
        }

        // Backward pass
        for y in (0..height).rev() {
            for x in (0..width).rev() {
                let p = uvec2(x as u32, y as u32);
                let mut d = *result.get(p);
                if d > 0.0 {
                    d = d.min(get(&result, x + 1, y) + 1.0);
                    d = d.min(get(&result, x, y + 1) + 1.0);
                    d = d.min(get(&result, x + 1, y + 1) + (2.0f32).sqrt());
                    d = d.min(get(&result, x - 1, y + 1) + (2.0f32).sqrt());
                    result.set(p, d);
                }
            }
        }

        result
    }

    pub fn distance_transform(&self) -> Img<f32> {
        let width = self.width;
        let height = self.data.len() as u32 / width;
        let mut dist = vec![0.0f32; (width * height) as usize];

        // Step 1: vertical EDT per column
        for x in 0..width {
            let mut f = vec![0f32; height as usize];
            for y in 0..height {
                f[y as usize] = if *self.get(uvec2(x, y)) == 0 {
                    0.0
                } else {
                    f32::INFINITY
                };
            }

            let d = edt_1d(&f);
            for y in 0..height {
                dist[(x + y * width) as usize] = d[y as usize];
            }
        }

        // Step 2: horizontal EDT per row
        for y in 0..height {
            let mut f = vec![0f32; width as usize];
            for x in 0..width {
                f[x as usize] = dist[(x + y * width) as usize];
            }

            let d = edt_1d(&f);
            for x in 0..width {
                dist[(x + y * width) as usize] = d[x as usize].sqrt();
            }
        }

        Img {
            width,
            data: dist,
        }
    }
}

fn edt_1d(f: &[f32]) -> Vec<f32> {
    let n = f.len();
    let mut d = vec![0f32; n];
    let mut v = vec![0usize; n]; // locations of parabolas in lower envelope
    let mut z = vec![-f32::INFINITY; n + 1]; // intersection locations
    let mut k = 0;

    v[0] = 0;
    z[0] = -f32::INFINITY;
    z[1] = f32::INFINITY;

    for q in 1..n {
        let mut s;
        loop {
            let p = v[k];
            s = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2. * (q - p) as f32);
            if s > z[k] {
                break;
            }
            k -= 1;
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f32::INFINITY;
    }

    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let dx = (q as i32 - v[k] as i32) as f32;
        d[q] = dx * dx + f[v[k]];
    }

    d
}