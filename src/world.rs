use std::cmp::PartialEq;
use std::time::Instant;
use fastnoise2::generator::*;
use fastnoise2::SafeNode;
use glam::{ivec3, uvec2, uvec3, vec2, vec3, UVec2, UVec3, Vec3, Vec3Swizzles};
use log::{info, warn};
use rand::{random, Rng};
use crate::utils::image::Img;
use crate::utils::image_transforms;
use crate::utils::sparse_bitmask::SparseBitmask;
use crate::utils::sparse_tree::{Node, SparseTree};
use crate::voxels::*;

pub struct World {
    /* Voxel data. The only data structure that is sent to the GPU. */
    pub data: SparseTree<Voxel>,

    /* Per-voxel metadata. Not sent to the GPU */
    is_generated: SparseBitmask, // Used to differentiate "air" voxels from "not generated" voxels
    // TODO: voxel entities, structures?
    // TODO: inverted bitmask for voxel out of the envelope

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
            is_generated: SparseBitmask::new(2 * depth as usize),
            seed,
            width: 4u32.pow(depth),
            depth,
            bottom_heightmap: Default::default(),
            top_heightmap: Default::default(),
        };
        let t = Instant::now();
        world.pre_generate_heightmaps();
        info!("Pre-generating heightmaps took {:?}", t.elapsed());
        let t = Instant::now();
        world.pre_generate_voxels();
        info!("Pre-generating voxels took {:?}", t.elapsed());
        let t = Instant::now();
        world.pre_generate_decoration();
        info!("Pre-generating decoration took {:?}", t.elapsed());
        let t = Instant::now();
        world.pre_generate_normals();
        info!("Pre-generating normals took {:?}", t.elapsed());
        world
    }

    pub(crate) fn raw_node_data(&self) -> &Vec<Node> { self.data.nodes.raw() }

    pub(crate) fn raw_voxel_data(&self) -> &Vec<Voxel> { self.data.voxels.raw() }

    pub fn peek(&self, coord: UVec3) -> u8 { todo!() }

    pub fn generate(&mut self, coord: UVec3) -> bool { todo!() }

    pub fn is_generated(&self, pos: UVec3) -> bool { self.is_generated.get(pos) }

    pub fn is_solid(&self, pos: UVec3) -> bool {
        if self.is_generated(pos) { self.data.get(pos).material() != Material::Air } else {
            pos.z <= *self.top_heightmap.get(pos.xy()) as u32 && pos.z >= *self.bottom_heightmap.get(pos.xy()) as u32
        }
    }
}

impl World {
    fn pre_generate_heightmaps(&mut self) {
        info!("width: {}", self.width);
        /* Generating a base heightmap */
        let tree_code = "IgAAAABAAACAPxoAARsAGwAZABkAEwDNzEw+DQAEAAAAAAAgQAkAAGZmJj8AAAAAPwAAAAAAAAAAgD8BHQAaAAAAAIA/ARwAAQUAAQAAAAAAAAAAAAAAAAAAAAAAAAAAMzPrQQAAAIA/AOxRGEAAMzMzQA==";
        let node = SafeNode::from_encoded_node_tree(tree_code).unwrap();
        let heightmap = Img::from_node(&node, self.width, 1024. / self.width as f32, self.seed);

        /* Using it to build a hard continent shape */
        let continent_shape = heightmap.map(|x, y, h| { if *h > -1. { 255u8 } else { 0u8 } });
        //continent_shape.to_file_and_show("continent_shape.png");

        /* Building the bottom envelope */
        let bottom_envelope = continent_shape.distance_transform().map(|_, _, h| { h * 2. });
        //bottom_envelope.map(|x, y, v| { (v / 200. * 255.) as u8 }).to_file_and_show("dist_to_sea.png");

        /* Building a very simple top envelope */
        let top_envelope = heightmap.map(|x, y, h| { (h + 1.) / 2. * 255. * self.width as f32 / 1024. });

        self.bottom_heightmap = bottom_envelope;
        self.top_heightmap = top_envelope;
    }

    fn pre_generate_voxels(&mut self) {
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
                    self.data.set(uvec3(x, y, z), if top_end - top_start > 1 { Material::Stone.into() } else { Material::Grass.into() });
                    self.is_generated.set(uvec3(x, y, z), true);
                }

                let bottom_start = center_z - depth;
                let bottom_end = center_z - min_neighbour_depth;

                for z in bottom_start..=bottom_end {
                    self.data.set(uvec3(x, y, z), Material::Stone.into());
                    self.is_generated.set(uvec3(x, y, z), true);
                }
            }
        }
    }

    fn pre_generate_decoration(&mut self) {
        let center_z = self.width / 2;
        let mut rng = rand::rng();
        for x in 0..self.width {
            for y in 0..self.width {
                let pos = uvec2(x, y);
                let height = *self.top_heightmap.get(pos) as u32;
                let pos = uvec3(x, y, center_z + height);
                if rng.random::<f64>() < 5e-4 && self.is_generated(pos) && self.data.get(pos).material() == Material::Grass {
                    // trunk
                    for i in 0..16 {
                        let target_pos = pos + uvec3(0, 0, i);
                        self.data.set(target_pos, Material::OakLog.into());
                        self.is_generated.set(target_pos, true);
                    }
                    // leaves
                    for i in 0..16i32 {
                        for j in 0..16i32 {
                            for k in 0..16i32 {
                                if (i - 8) * (i - 8) + (j - 8) * (j - 8) + (k - 8) * (k - 8) <= 8 * 8 {
                                    let target_pos = uvec3(pos.x + i as u32 - 8, pos.y + j as u32 - 8, pos.z + k as u32 + 12);
                                    if !self.is_solid(target_pos) {
                                        self.data.set(target_pos, Material::OakLeaves.into());
                                        self.is_generated.set(target_pos, true);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn pre_generate_normals(&mut self) {
        let mut copy = self.data.clone();
        let mut count = 0u32;
        self.data.for_each(|pos, voxel| {
            let radius = 2;
            let mut normal = Vec3::default();
            let category = self.data.get(pos).material().category();
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    for dz in -radius..=radius {
                        if dx == 0 && dy == 0 && dz == 0 { continue; }
                        let offset = ivec3(dx, dy, dz);
                        let local_pos = (pos.as_ivec3() + offset).as_uvec3();
                        let local_height = *self.top_heightmap.get(local_pos.xy()) as u32;
                        let local_depth = *self.bottom_heightmap.get(local_pos.xy()) as u32;
                        let is_solid = {
                            if self.is_generated(local_pos) {
                                self.data.get(local_pos).material().category() == category
                            } else if category == MaterialCategory::Terrain {
                                (local_pos.z <= self.width / 2 + local_height)
                                    && (local_pos.z >= self.width / 2 - local_depth) && local_height != 0 && local_depth != 0
                            } else { false }
                        };
                        let distance = offset.as_vec3().length();
                        if !is_solid && distance > 0.0 {
                            let weight = 1.0 / (distance);
                            normal += offset.as_vec3().normalize() * weight;
                        }
                    }
                }
            }
            let mut voxel = voxel.clone();
            voxel.set_normal(normal.normalize());
            copy.set(pos, voxel);
            count += 1;
        });
        info!("Generated {} voxels !", count);
        self.data = copy;
    }
}