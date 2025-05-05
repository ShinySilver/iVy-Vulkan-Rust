use fastnoise2::generator::GeneratorWrapper;
use fastnoise2::SafeNode;
use glam::UVec3;

use crate::contree::Contree;

type Voxel = u8;

#[derive(Default, Copy, Clone, PartialEq)]
struct ChunkMeta {
    is_generated: u64,
}

#[derive(Default)]
struct MetadataStorage {
    metadata: Contree<ChunkMeta>,
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

#[derive(Default)]
pub struct World {
    /* Voxel data */
    data: Contree<Voxel>,

    /* Voxel metadata */
    metadata: MetadataStorage,

    /* World planning */
}

impl World {}

pub trait WorldGenerator{

}