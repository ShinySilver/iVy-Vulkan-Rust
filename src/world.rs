use crate::utils::image::Img;
use crate::utils::sparse_bitmask::SparseBitmask;
use crate::utils::sparse_tree::{Node, SparseTree};
use crate::voxels::*;
use fast_poisson::Poisson2D;
use fastnoise2::generator::*;
use fastnoise2::SafeNode;
use glam::{ivec2, ivec3, uvec2, uvec3, vec2, vec3, IVec2, IVec3, UVec2, UVec3, Vec3, Vec3Swizzles};
use log::{info, warn};
use rand::distr::Uniform;
use rand::{random, Rng};
use std::cmp::PartialEq;
use std::ops::{Add, Div, Mul};
use std::time::Instant;

pub struct World {
    /* Voxel data. The only data structure that is sent to the GPU. */
    pub data: SparseTree<Voxel>,

    /* Per-voxel metadata. Not sent to the GPU */
    is_generated: SparseBitmask, // Used to differentiate "air" voxels from "not generated" voxels

    /* World metadata */
    pub time: f32,
    start_time: Instant,
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
            time: 0.0,
            start_time: Instant::now(),
            seed,
            width: 4u32.pow(depth),
            depth,
            bottom_heightmap: Default::default(),
            top_heightmap: Default::default(),
        };
        let t = Instant::now();
        world.generate_heightmaps();
        info!("Pre-generating heightmaps took {:?}", t.elapsed());
        let t = Instant::now();
        world.generate_terrain_voxels();
        info!("Pre-generating terrain took {:?}", t.elapsed());
        let t = Instant::now();
        world.generate_terrain_normals();
        info!("Pre-generating terrain normals took {:?}", t.elapsed());
        let t = Instant::now();
        world.generate_decoration_voxels();
        info!("Pre-generating decoration took {:?}", t.elapsed());
        let t = Instant::now();
        world.generate_decoration_normals();
        info!("Pre-generating decoration normals took {:?}", t.elapsed());
        world
    }

    pub fn update(&mut self) {
        self.time = self.start_time.elapsed().as_secs_f32();
    }

    pub(crate) fn raw_node_data(&self) -> &Vec<Node> { self.data.nodes.raw() }

    pub(crate) fn raw_voxel_data(&self) -> &Vec<Voxel> { self.data.voxels.raw() }

    pub fn peek(&self, coord: UVec3) -> u8 { todo!() }

    pub fn generate(&mut self, coord: UVec3) -> bool { todo!() }

    pub fn is_solid(&self, pos: UVec3) -> bool {
        if self.is_generated.get(pos) { self.data.get(pos).material() != Material::Air } else {
            pos.z <= *self.top_heightmap.get(pos.xy()) as u32 && pos.z >= *self.bottom_heightmap.get(pos.xy()) as u32
        }
    }
}

impl World {
    fn generate_heightmaps(&mut self) {
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

    fn generate_heightmaps_v2(&mut self) {
        info!("width: {}", self.width);

        let scaling_factor = 1024. / self.width as f32;

        /* Placeholder noise */
        let tree_code = "AAAAAIA/";
        let placeholder_node = SafeNode::from_encoded_node_tree(tree_code).unwrap();

        /* Generating the continent shape */
        let tree_code = "HQAeABoAARsAGwAZABMAzcxMPg0ABQAAAAAAIEAJAABmZiY/AAAAAD8AAACAPwEdABoAAAAAgD8BHAABBQABAAAAAAAAAAAAAAAAAAAAAAAAAAAzM+tBAAAAgD8A7FEYQAAzMzNAAAAAgL8AzcxMvw==";
        let node = SafeNode::from_encoded_node_tree(tree_code).unwrap();
        let continent_shape = Img::from_node(&node, self.width, scaling_factor, self.seed);

        /* Generating a simple hilly noise */
        let tree_code = "CQA=";
        let node = SafeNode::from_encoded_node_tree(tree_code).unwrap();
        let hilly_noise = Img::from_node(&node, self.width, scaling_factor, self.seed);

        /* Generating a simple mountainous noise */
        let tree_code = "";
        let node = SafeNode::from_encoded_node_tree(tree_code).unwrap_or_else(|_| { placeholder_node });
        let mountainous_noise = Img::from_node(&node, self.width, scaling_factor, self.seed);

        /* Continental meta noise */
        let tree_code = "JQDNzMw+zczMPs3MzD4AAIA/CQA=";
        let node = SafeNode::from_encoded_node_tree(tree_code).unwrap();
        let continental_noise = Img::from_node(&node, self.width, scaling_factor, self.seed);

        /* Building the top envelope */
        let easing_width = 12. / scaling_factor;
        let easing_factor = 2.2;
        let top_envelope = continent_shape
            .map(|x, y, h| { if *h > -1. { 255u8 } else { 0u8 } })
            .distance_transform()
            .map(|x, y, h| { h.clamp(0.0, easing_width).div(easing_width).powf(1. / easing_factor) })
            .map(|x, y, h| { h.max(0.) });

        /* Assembling the top heightmap */
        let hills_height = 18.;
        self.top_heightmap = top_envelope
            .map(|x, y, h| { h * hilly_noise.get(uvec2(x, y)).add(1.).mul(0.5) })
            .map(|_, _, h| h * hills_height / scaling_factor);

        /* Building the bottom envelope */
        let easing_width = 22. / scaling_factor;
        let easing_height = 12. / scaling_factor;
        let easing_factor = 2.2;
        let lambda = 2e-4;
        let bottom_envelope = continent_shape
            .map(|x, y, h| { if *h > -1. { 255u8 } else { 0u8 } })
            .distance_transform()
            .map(|_, _, h| {
                (h.clamp(0.0, easing_width)
                    .div(easing_width).powf(1. / easing_factor)
                    .max(0.) * easing_height - h * 2.2)
                    .mul((-lambda * h.mul(scaling_factor).powf(1.7)).exp()) + h * 2.2
            });

        self.bottom_heightmap = bottom_envelope;
    }

    fn generate_terrain_voxels(&mut self) {
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
                        min_neighbour_height = min_neighbour_height.min(n_height);
                        let n_depth = *self.bottom_heightmap.get(uvec2(nx, ny)) as u32;
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
                    self.data.set(uvec3(x, y, z), if top_end - top_start > 3 { Material::Stone.into() } else { Material::Grass.into() });
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

    fn generate_terrain_normals(&mut self) {
        let mut count = 0u32;
        const RADIUS: i32 = 5;
        const WIDTH: usize = (RADIUS * 2 + 1) as usize;
        let kernel: [IVec2; WIDTH.pow(2)] = std::array::from_fn(|i| {
            let dx = (i / WIDTH) as i32 - RADIUS;
            let dy = (i % WIDTH) as i32 - RADIUS;
            IVec2::new(dx, dy)
        });
        self.data.for_each_mut(|pos, voxel| {
            let mut normal = Vec3::default();
            for &offset in &kernel {
                let local_pos_2d = pos.xy().wrapping_add(uvec2(offset.x as u32, offset.y as u32));
                let local_height = self.width / 2 + *self.top_heightmap.get(local_pos_2d) as u32;
                let local_depth = self.width / 2 - *self.bottom_heightmap.get(local_pos_2d) as u32;
                for dz in (-RADIUS..=RADIUS) {
                    if offset.x == 0 && offset.y == 0 && dz == 0 { continue; }
                    let height = (pos.z as i32 + dz) as u32;
                    if height <= local_height && height >= local_depth { continue; }
                    let offset = ivec3(offset.x, offset.y, dz).as_vec3();
                    let weight = 1.0 / offset.length().powi(2);
                    normal += offset * weight;
                }
            }
            voxel.set_normal(normal.normalize());
            count += 1;
        });
    }

    fn generate_decoration_voxels(&mut self) {
        let tree_spacing = 14;
        let center_z = self.width / 2;
        let mut sampler = Poisson2D::new()
            .with_dimensions([self.width as f64; 2], tree_spacing as f64)
            .with_seed(self.seed as u64);
        let points = sampler.generate()
            .iter()
            .map(|&p| uvec2(p[0] as u32, p[1] as u32))
            .collect::<Vec<UVec2>>();
        let rng = &mut rand::rng();

        for base in points {
            let ground_h = *self.top_heightmap.get(base) as u32;
            let base_pos = uvec3(base.x, base.y, center_z + ground_h);
            if !self.is_generated.get(base_pos) || self.data.get(base_pos).material() != Material::Grass {
                continue;
            }
            // 1) Trunk
            let trunk_height = rng.random_range(10..=16);
            for dz in 0..trunk_height {
                let pos = base_pos + uvec3(0, 0, dz);
                self.data.set(pos, Material::OakLog.into());
                self.is_generated.set(pos, true);
            }

            // 2) Branches
            let branch_count = rng.random_range(2..=5);
            for _ in 0..branch_count {
                let start_h = rng.random_range((trunk_height as f32 * 0.3) as u32..(trunk_height as f32 * 0.8) as u32);
                let angle = rng.random_range(0.0..std::f32::consts::TAU);
                let length = rng.random_range(4..=8) as i32;
                let tilt = rng.random_range(0.2..0.6); // upward pitch
                let mut current = base_pos + uvec3(0, 0, start_h);
                for i in 0..length {
                    // move horizontally
                    let dx = (angle.cos() * (i as f32)) as i32;
                    let dy = (angle.sin() * (i as f32)) as i32;
                    // small upward step
                    let dz = ((i as f32) * tilt) as i32;
                    current = uvec3(
                        (base.x as i32 + dx) as u32,
                        (base.y as i32 + dy) as u32,
                        center_z + ground_h + start_h + dz as u32,
                    );
                    // place the branch log
                    if !self.is_generated.get(current) {
                        self.data.set(current, Material::OakLog.into());
                        self.is_generated.set(current, true);
                    }
                }

                // leaf cluster at end
                let cluster_radius = rng.random_range(2..=3) as i32;
                let shift = |c: u32, d: i32| {
                    if d >= 0 { c.checked_add(d as u32) } else { c.checked_sub((-d) as u32) }
                };

                for x in -cluster_radius..=cluster_radius {
                    for y in -cluster_radius..=cluster_radius {
                        for z in -cluster_radius..=cluster_radius {
                            if x * x + y * y + z * z <= cluster_radius * cluster_radius {
                                //info!("L");
                                // try all three axes at once
                                if let (Some(nx), Some(ny), Some(nz)) = (
                                    shift(current.x, x),
                                    shift(current.y, y),
                                    shift(current.z, z),
                                ) {
                                    let leaf_pos = uvec3(nx, ny, nz);
                                    //info!("M");
                                    if !self.is_solid(leaf_pos) {
                                        self.data.set(leaf_pos, Material::OakLeaves.into());
                                        self.is_generated.set(leaf_pos, true);
                                    }
                                } else {
                                    info!("skip leaf overflow at {:?}", (x, y, z));
                                }
                            }
                        }
                    }
                }
            }

            // 3) Canopy: an ellipsoid around the top
            let shift = |c: u32, d: i32| {
                if d >= 0 { c.checked_add(d as u32) } else { c.checked_sub((-d) as u32) }
            };

            let canopy_h = trunk_height + 2;
            let rx = 8;
            let ry = 8;
            let rz = 5;
            let center = base_pos + uvec3(0, 0, canopy_h);

            for dx in -rx..=rx {
                for dy in -ry..=ry {
                    for dz in -rz..=rz {
                        let fx = dx as f32 / rx as f32;
                        let fy = dy as f32 / ry as f32;
                        let fz = dz as f32 / rz as f32;
                        if fx * fx + fy * fy + fz * fz <= 1.0 {
                            // try shifting each axis safely
                            if let (Some(nx), Some(ny), Some(nz)) = (
                                shift(center.x, dx),
                                shift(center.y, dy),
                                shift(center.z, dz),
                            ) {
                                let leaf_pos = uvec3(nx, ny, nz);
                                if !self.is_solid(leaf_pos) {
                                    self.data.set(leaf_pos, Material::OakLeaves.into());
                                    self.is_generated.set(leaf_pos, true);
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    fn generate_decoration_normals(&mut self) {
        let mut copy = self.data.clone();
        let mut count = 0u32;
        self.data.for_each(|pos, voxel| {
            let radius = 2;
            let mut normal = Vec3::default();
            let category = self.data.get(pos).material().category();
            if category == MaterialCategory::Terrain { return; }
            let mut has_adjacent_air = false;
            for neighbour in [
                ivec3(-1, 0, 0),
                ivec3(0, -1, 0),
                ivec3(0, 0, -1),
                ivec3(1, 0, 0),
                ivec3(0, 1, 0),
                ivec3(0, 0, 1)] {
                if !self.is_generated.get((pos.as_ivec3() + neighbour).as_uvec3()) {
                    has_adjacent_air = true;
                    break;
                }
            }
            if !has_adjacent_air { return; }
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    for dz in -radius..=radius {
                        if dx == 0 && dy == 0 && dz == 0 { continue; }
                        let offset = ivec3(dx, dy, dz);
                        let local_pos = (pos.as_ivec3() + offset).as_uvec3();
                        let local_height = *self.top_heightmap.get(local_pos.xy()) as u32;
                        let local_depth = *self.bottom_heightmap.get(local_pos.xy()) as u32;
                        let is_solid = {
                            if self.is_generated.fast_unsafe_lookup(local_pos) {
                                self.data.get(local_pos).material().category() == category
                            } else if category == MaterialCategory::Terrain {
                                (local_pos.z <= self.width / 2 + local_height)
                                    && (local_pos.z >= self.width / 2 - local_depth) && local_height != 0 && local_depth != 0
                            } else { false }
                        };
                        let distance = offset.as_vec3().length();
                        if !is_solid && distance > 0.0 {
                            let weight = 1.0 / distance.sqrt();
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
        self.data = copy;
    }
}