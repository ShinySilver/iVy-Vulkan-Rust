use glam::UVec3;
use crate::utils::memory_pool::MemoryPool;

#[derive(Copy, Clone, Default)]
struct Node {
    child_uniform: u8,  // Bit = 1 → child is uniform
    child_type: u8,     // Bit value when child is uniform
    child_index: usize, // Index into memory pool (non-uniform only)
}

#[derive(Default)]
pub struct SparseBitmask {
    tree_depth: usize,
    root: Node,
    nodes: MemoryPool<Node>,
    leaves: MemoryPool<u8>,
}


impl SparseBitmask {
    pub fn new(tree_depth: usize) -> Self {
        Self {
            tree_depth,
            root: Node::default(),
            nodes: MemoryPool::new(1024),
            leaves: MemoryPool::new(1024),
        }
    }

    pub fn set(&mut self, mut pos: UVec3, value: bool) {}

    pub fn get(&self, mut pos: UVec3) -> bool { false }

    pub fn get_mut(&mut self, _pos: UVec3) -> Option<&mut bool> { None }
}


#[cfg(test)]
mod tests {
    use super::*;
    use glam::uvec3;

    fn setup_bitmask(depth: usize) -> SparseBitmask {
        SparseBitmask::new(depth)
    }

    #[test]
    fn test_default_is_false() {
        let mask = setup_bitmask(4);
        assert!(!mask.get(uvec3(0, 0, 0)));
        assert!(!mask.get(uvec3(7, 7, 7)));
        assert!(!mask.get(uvec3(15, 15, 15))); // max in 4-level tree
    }

    #[test]
    fn test_set_and_get_true() {
        let mut mask = setup_bitmask(4);
        let pos = uvec3(3, 5, 7);
        mask.set(pos, true);
        assert!(mask.get(pos));
    }

    #[test]
    fn test_set_and_get_false() {
        let mut mask = setup_bitmask(4);
        let pos = uvec3(3, 5, 7);
        mask.set(pos, true);
        mask.set(pos, false);
        assert!(!mask.get(pos));
    }

    #[test]
    fn test_multiple_positions() {
        let mut mask = setup_bitmask(4);
        let positions = [
            uvec3(0, 0, 0),
            uvec3(1, 2, 3),
            uvec3(7, 7, 7),
            uvec3(15, 0, 0),
            uvec3(0, 15, 15),
        ];

        for &pos in &positions {
            mask.set(pos, true);
        }

        for &pos in &positions {
            assert!(mask.get(pos), "Expected true at {:?}", pos);
        }

        assert!(!mask.get(uvec3(8, 8, 8))); // untouched
        assert!(!mask.get(uvec3(14, 14, 14))); // untouched
    }

    #[test]
    fn test_overwrite_different_values() {
        let mut mask = setup_bitmask(4);
        let pos = uvec3(6, 6, 6);

        mask.set(pos, false);
        assert!(!mask.get(pos));

        mask.set(pos, true);
        assert!(mask.get(pos));

        mask.set(pos, false);
        assert!(!mask.get(pos));
    }

    #[test]
    fn test_non_interference_between_voxels() {
        let mut mask = setup_bitmask(4);
        let a = uvec3(1, 1, 1);
        let b = uvec3(14, 14, 14);

        mask.set(a, true);
        assert!(mask.get(a));
        assert!(!mask.get(b));

        mask.set(b, true);
        assert!(mask.get(a));
        assert!(mask.get(b));

        mask.set(a, false);
        assert!(!mask.get(a));
        assert!(mask.get(b));
    }
}
