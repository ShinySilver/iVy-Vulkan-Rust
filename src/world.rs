use std::cmp::max;
use std::ops::*;
use std::time::Instant;
use fastnoise2::generator::{Generator, GeneratorWrapper};
use fastnoise2::generator::prelude::opensimplex2;
use fastnoise2::SafeNode;
use glam::{uvec2, vec2, UVec2, UVec3, Vec2};
use half::f16;
use log::{info, warn};
use crate::sparse_tree::SparseTree;
use crate::worldgen::grid::Grid;
use crate::worldgen::image::ImageF16;

type Voxel = u8;

#[derive(Default, Copy, Clone, PartialEq)]
struct ChunkMeta {
    is_generated: u64,
}

#[derive(Default)]
struct MetadataStorage {
    metadata: SparseTree<ChunkMeta>,
}

impl MetadataStorage {
    fn is_generated(&self, pos: UVec3) -> bool {
        let chunk_coords = pos / 4;
        let bit_index = (pos % UVec3::new(1, 4, 16)).element_sum();
        self.metadata.get(chunk_coords).is_generated & (0x1u64 << bit_index) != 0
    }

    fn set_generated(&mut self, pos: UVec3) {
        let chunk_coords = pos / 4;
        let bitmask = 0x1u64 << (pos % UVec3::new(1, 4, 16)).element_sum();
        match self.metadata.get_mut(chunk_coords) {
            None => { self.metadata.set(pos, ChunkMeta { is_generated: bitmask }); }
            Some(chunk_meta) => { chunk_meta.is_generated |= bitmask; }
        }
    }
}

#[derive(Default, Clone)]
struct Biome {
    height: f32,
}

pub struct World {
    /* Voxel data */
    data: SparseTree<Voxel>,

    /* Voxel metadata */
    metadata: MetadataStorage,

    /* World metadata */
    seed: u64,
    width: u32,
    depth: u32,

    /* World planning */
    grid: Grid<Biome>,
}

impl World {
    pub fn new(depth: u32, seed: u64) -> Self {
        let mut world = World {
            data: Default::default(),
            metadata: Default::default(),
            seed,
            width: 4u32.pow(depth),
            depth,
            grid: Grid::new(4u32.pow(depth), (4u32.pow(depth) / 160u32), seed),
        };
        world.plan();
        world
    }
    fn plan(&mut self) {
        /* Generating a base heightmap */

        const SEED: i32 = 145896;
        for i in 0..20 {
            let heightmap = {
                let mut world_heightmap = vec![0.0; self.width.pow(2) as usize];
                let tree_code = "IgAAAABAAACAPxoAARsAGwAZABkAEwDNzEw+DQAEAAAAAAAgQAkAAGZmJj8AAAAAPwAAAAAAAAAAgD8BHQAaAAAAAIA/ARwAAQUAAQAAAAAAAAAAAAAAAAAAAAAAAAAAMzPrQQAAAIA/AOxRGEAAMzMzQA==";
                let node = SafeNode::from_encoded_node_tree(tree_code).unwrap();

                /**
                 * Best seeds:
                 *   145892
                 *   145894
                 *   ... ?
                 */
                let min_max = node.gen_uniform_grid_2d(&mut world_heightmap,
                                                       self.width as i32 / -2, self.width as i32 / -2,
                                                       self.width as i32, self.width as i32, 0.8e-2, SEED+i);
                ImageF16 {
                    width: self.width,
                    data: world_heightmap.iter().map(|h| { f16::from_f32((h + 1.) / 2. * 255.) }).collect(),
                }
            };
            let path = format!("heightmap_{}.png", i);
            heightmap.to_u8(0., 1.).export(&path);
        }
        opener::open("heightmap_1.png").unwrap_or_else(|_| warn!("Failed to open {}", "heightmap_1.png"));

        /*
        /* Applying the heightmap to the biome map */
        self.grid.apply(|i, pos: &UVec2, biome: &mut Biome| {
            biome.height = heightmap.get(pos).to_f32()
        });
        self.grid.export("biome_heightmap_1.png", |biome| { [biome.height as u8; 3] });

        /* Apply a smooth circular mask */
        self.grid.apply(|i, pos: &UVec2, biome: &mut Biome| {
            let center_pos = pos.as_vec2() - self.width as f32 / 2.;
            biome.height *= f32::min(1., 1. - 0.2 * center_pos.dot(center_pos) / (self.width / 4).pow(2) as f32);
        });
        self.grid.export("biome_heightmap_2.png", |biome| { [biome.height as u8; 3] });

        /* Apply a threshold to get the continent shape */
        let threshold = 50.;
        self.grid.apply(|i, pos: &UVec2, biome: &mut Biome| {
            if biome.height < threshold { biome.height = 0.0; }
        });
        self.grid.export("biome_heightmap_3.png", |biome| {
            if biome.height != 0. { [biome.height as u8; 3] } else { [46, 139, 187] }
        });*/
    }
    fn generate(&mut self, coord: [f64; 2]) -> bool { false }
}