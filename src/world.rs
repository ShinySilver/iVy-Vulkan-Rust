use fastnoise2::generator::*;
use fastnoise2::SafeNode;
use glam::{uvec2, uvec3, vec2, UVec2, UVec3};
use log::{info, warn};
use crate::utils::image::Img;
use crate::utils::image_transforms;
use crate::utils::sparse_bitmask::SparseBitmask;
use crate::utils::sparse_tree::{Node, SparseTree};

#[repr(u8)]
enum Voxel {
    Air,
    DebugRed,
    DebugGreen,
    DebugBlue,
    Stone,
    Dirt,
    Grass,
}

pub struct World {
    /* Voxel data. The only data structure that is sent to the GPU. */
    pub data: SparseTree<u16>,

    /* Per-voxel metadata. Not sent to the GPU */
    is_generated: SparseBitmask, // Used to differentiate "air" voxels from "not generated" voxels
    // TODO: voxel entities, structures?

    /* World metadata */
    seed: i32,
    width: u32,
    depth: u32,

    /* World planning */
    bottom_heightmap: Img<f32>,
    top_heightmap: Img<f32>,
}

impl World {
    pub fn new(depth: u32, seed: i32) -> Self {
        let mut world = World {
            data: SparseTree::new(depth as usize),
            is_generated: Default::default(),
            seed,
            width: 4u32.pow(depth),
            depth,
            bottom_heightmap: Default::default(),
            top_heightmap: Default::default(),
        };
        world.plan();
        world.pre_generate();
        world
    }

    pub(crate) fn raw_voxel_data(&self) -> &Vec<Node> { self.data.nodes.raw() }

    fn plan(&mut self) {
        info!("width: {}", self.width);
        /* Generating a base heightmap */
        let tree_code = "IgAAAABAAACAPxoAARsAGwAZABkAEwDNzEw+DQAEAAAAAAAgQAkAAGZmJj8AAAAAPwAAAAAAAAAAgD8BHQAaAAAAAIA/ARwAAQUAAQAAAAAAAAAAAAAAAAAAAAAAAAAAMzPrQQAAAIA/AOxRGEAAMzMzQA==";
        let node = SafeNode::from_encoded_node_tree(tree_code).unwrap();
        let heightmap = Img::from_node(&node, self.width, self.seed);

        /* Using it to build a hard continent shape */
        let continent_shape = heightmap.map(|x, y, h| { if *h > -1. { 255u8 } else { 0u8 } });
        //continent_shape.to_file_and_show("continent_shape.png");

        /* Building the bottom envelope */
        let bottom_envelope = continent_shape.distance_transform().map(|_, _, h| { h * 2. });
        //bottom_envelope.map(|x, y, v| { (v / 200. * 255.) as u8 }).to_file_and_show("dist_to_sea.png");

        /* Building a very simple top envelope */
        let top_envelope = heightmap.map(|x, y, h| { (h + 1.) / 2. * 255. });

        self.bottom_heightmap = bottom_envelope;
        self.top_heightmap = top_envelope;
    }

    fn pre_generate(&mut self) {
        let center_z = self.width / 2;

        for x in 0..self.width {
            for y in 0..self.width {
                let pos = uvec2(x, y);
                let height = *self.top_heightmap.get(pos) as u32;
                let depth = *self.bottom_heightmap.get(pos) as u32;

                // checking the min height and depth of the neighbors (x±1, y±1)
                let mut min_neighbour_height = height;
                let mut min_neighbour_depth = depth;
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = x.wrapping_add(dx as u32);
                    let ny = y.wrapping_add(dy as u32);
                    if nx < self.width && ny < self.width {
                        let n_height = *self.top_heightmap.get(uvec2(nx, ny)) as u32;
                        let n_depth = *self.bottom_heightmap.get(uvec2(nx, ny)) as u32;
                        min_neighbour_height = min_neighbour_height.min(n_height);
                        min_neighbour_depth = min_neighbour_depth.min(n_depth);
                    }
                }

                // No filling the void with a thin layer of stone
                if height == 0 && depth == 0 && min_neighbour_height == 0 && min_neighbour_depth == 0 {
                    continue; // skip this column entirely
                }

                // Fill from neighbor-based depth/height to current point to avoid holes
                let top_start = center_z + min_neighbour_height;
                let top_end = center_z + height;

                for z in top_start..=top_end {
                    self.data.set(uvec3(x, y, z), Voxel::Grass as u16);
                }

                let bottom_start = center_z - depth;
                let bottom_end = center_z - min_neighbour_depth;

                for z in bottom_start..=bottom_end {
                    self.data.set(uvec3(x, y, z), Voxel::Stone as u16);
                }
            }
        }
    }

    pub fn peek(&self, coord: UVec3) -> u8 { todo!() }

    pub fn generate(&mut self, coord: UVec3) -> bool { todo!() }
}