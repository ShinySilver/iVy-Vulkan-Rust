use fast_poisson::{Poisson2D, Point};
use glam::{uvec2, UVec2};
use kiddo::{KdTree, SquaredEuclidean};
use image::{Rgb, RgbImage};
use log::warn;

#[derive(Clone)]
pub struct Grid<T: Default + Clone> {
    width: u32,
    points: Vec<UVec2>,
    data: Vec<T>,
    kdtree: KdTree<f32, 2>,
}

impl<T: Default + Clone> Grid<T> {
    pub fn new(width: u32, radius: u32, seed: u64) -> Self {
        let mut sampler = Poisson2D::new().with_dimensions([width as f64; 2], radius as f64).with_seed(seed);
        let points = sampler.generate().iter().map(|&p| uvec2(p[0] as u32, p[1] as u32)).collect::<Vec<UVec2>>();
        let data = vec![T::default(); points.len()];
        let mut kdtree = KdTree::new();
        for (i, p) in points.iter().enumerate() {
            kdtree.add(&[p.x as f32, p.y as f32], i as u64);
        }
        Grid {
            width,
            points,
            data,
            kdtree,
        }
    }

    pub fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &UVec2, &mut T),
    {
        for (i, (p, v)) in self.points.iter().zip(self.data.iter_mut()).enumerate() {
            f(i, p, v);
        }
    }

    pub fn get(&self, index: usize) -> (&UVec2, &T) {
        assert!(index < self.points.len());
        (&self.points[index], &self.data[index])
    }

    pub fn set(&mut self, index: usize, value: T) {
        assert!(index < self.data.len());
        self.data[index] = value;
    }

    pub fn k_nearest(&self, coord: UVec2, k: usize) -> Vec<(&UVec2, usize)> {
        self.kdtree
            .nearest_n::<SquaredEuclidean>(&[coord.x as f32, coord.y as f32], k)
            .into_iter()
            .map(|neighbour| (&self.points[neighbour.item as usize], neighbour.item as usize))
            .collect()
    }

    pub fn nearest(&self, coord: UVec2) -> (&UVec2, usize) {
        let index = self.kdtree.nearest_one::<SquaredEuclidean>(&[coord.x as f32, coord.y as f32]).item as usize;
        (&self.points[index], index)
    }

    pub fn export<F>(&self, title: &str, color_fn: F) -> RgbImage
    where
        F: Fn(&T) -> [u8; 3],
    {
        let img = RgbImage::from_fn(self.width, self.width, |x, y| {
            let (_, index) = self.nearest(uvec2(x, y));
            Rgb(color_fn(&self.data[index]))
        });
        img.save(title).unwrap_or_else(|_| warn!("Failed to save {}", title));
        //opener::open(title).unwrap_or_else(|_| warn!("Failed to open {}", title));
        img
    }

    pub fn k_export<F>(&self, title: &str, color_fn: F) -> RgbImage
    where
        F: Fn(&UVec2, &Vec<(&UVec2, &T)>) -> [u8; 3],
    {
        let img = RgbImage::from_fn(self.width, self.width, |x, y| {
            let k_nearest = self.k_nearest(uvec2(x, y), 6);
            Rgb(color_fn(&uvec2(x, y), &k_nearest.iter().map(|&(pos, index)| (pos, &self.data[index])).collect()))
        });
        img.save(title).unwrap_or_else(|_| warn!("Failed to save {}", title));
        //opener::open(title).unwrap_or_else(|_| warn!("Failed to open {}", title));
        img
    }
}