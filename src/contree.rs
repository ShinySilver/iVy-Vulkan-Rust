use crate::mempool::MemoryPool;
use std::fmt::Display;
use glam::UVec3;

#[repr(C)]
#[derive(Default, Copy, Clone)]
struct Node {
    bitmask: u64,
    data: u32,
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.data & (0b1u32 << 31) == 0
    }
    fn children_index(&self) -> usize {
        (self.data & !(0b1u32 << 31)) as usize
    }
}

pub struct ConTree<T: Default + Copy + PartialEq + Display> {
    nodes: MemoryPool<Node>,
    voxels: MemoryPool<T>,
    root_index: usize,
    tree_depth: usize,
}

impl<T: Default + Copy + PartialEq + Display> ConTree<T> {
    pub fn new(tree_depth: usize) -> Self {
        assert!(tree_depth > 0, "Invalid tree_depth");
        let mut nodes = MemoryPool::default();
        let (_, root_index) = nodes.allocate();
        ConTree {
            nodes,
            voxels: MemoryPool::default(),
            root_index,
            tree_depth,
        }
    }

    pub fn set_voxel(&mut self, pos: UVec3, voxel: T) {
        if voxel == Default::default() {
            self.remove_voxel(pos);
            return;
        }
        let (_, mut stack) = self.find_node(pos);
        while stack.len() != self.tree_depth {
            let child_width = 0x1u32 << (self.tree_depth - stack.len()) * 2;
            let child_local_pos = (pos & (child_width * 4 - 1)) / child_width;
            let child_xyz = child_local_pos.dot(UVec3::new(1, 4, 16));
            let (_, index) = self.create_node_child(*stack.last().unwrap(), child_xyz);
            stack.push(index);
        }
        let target_node_index = *stack.last().unwrap();
        let child_local_pos = pos & 0b11u32;
        let child_xyz = child_local_pos.dot(UVec3::new(1, 4, 16));
        {
            let node = self.nodes.acquire_mut(target_node_index);
            if node.bitmask & (0x1u64 << child_xyz) != 0 {
                let child_local_index = (node.bitmask & !(u64::MAX << child_xyz)).count_ones() as usize;
                let target = self.voxels.acquire_mut(node.children_index() + child_local_index);
                *target = voxel;
                println!();
                return;
            }
        }
        let (new_voxel, _) = self.create_voxel_child(target_node_index, child_xyz);
        *new_voxel = voxel;
        println!();
    }

    pub fn get_voxel(&self, pos: UVec3) -> T {
        let (opt, _) = self.find_node(pos);
        match opt {
            Some(node) => {
                let child_local_pos = pos & 0b11u32;
                let child_xyz = child_local_pos.dot(UVec3::new(1, 4, 16));
                if (node.bitmask & (0x1u64 << child_xyz)) == 0 {
                    return Default::default();
                }

                let child_local_index = (node.bitmask & !(u64::MAX << child_xyz)).count_ones() as usize;
                let voxel_index = node.children_index() + child_local_index;
                *self.voxels.acquire(voxel_index)
            }
            None => Default::default(),
        }
    }

    pub fn remove_voxel(&mut self, _pos: UVec3) {
        unimplemented!()
    }

    fn create_node_child(&mut self, parent_index: usize, child_xyz: u32) -> (&mut Node, usize) {
        let parent = *self.nodes.acquire(parent_index);
        assert_eq!(parent.bitmask & (0x1u64 << child_xyz), 0, "Overriding an existing child node");
        let previous_child_count = parent.bitmask.count_ones();
        let new_child_local_index = (parent.bitmask & !(u64::MAX << child_xyz)).count_ones();
        let (_, new_child_array_index) = self.nodes.allocate_multiple((previous_child_count + 1) as usize);
        let new_child_index = new_child_array_index + new_child_local_index as usize;
        for i in 0..new_child_local_index as usize {
            *self.nodes.acquire_mut(new_child_array_index + i) = *self.nodes.acquire(parent.children_index() + i);
        }
        *self.nodes.acquire_mut(new_child_index) = Default::default();
        for i in new_child_local_index..previous_child_count {
            *self.nodes.acquire_mut(new_child_array_index + i as usize + 1) =
                *self.nodes.acquire(parent.children_index() + i as usize);
        }
        {
            let parent = self.nodes.acquire_mut(parent_index);
            parent.bitmask |= 0x1u64 << child_xyz;
            parent.data = parent.data & (0x1u32 << 31) | (new_child_array_index as u32);
        }
        (self.nodes.acquire_mut(new_child_index), new_child_index)
    }

    fn create_voxel_child(&mut self, parent_index: usize, child_xyz: u32) -> (&mut T, usize) {
        let parent = *self.nodes.acquire(parent_index);
        let previous_child_count = parent.bitmask.count_ones();
        let new_child_local_index = (parent.bitmask & !(u64::MAX << child_xyz)).count_ones();
        let (_, new_child_array_index) = self.voxels.allocate_multiple((previous_child_count + 1) as usize);
        let new_child_index = new_child_array_index + new_child_local_index as usize;
        for i in 0..new_child_local_index as usize {
            *self.voxels.acquire_mut(new_child_array_index + i) = *self.voxels.acquire(parent.children_index() + i);
        }
        *self.voxels.acquire_mut(new_child_index) = Default::default();
        for i in new_child_local_index..previous_child_count {
            *self.voxels.acquire_mut(new_child_array_index + i as usize + 1) =
                *self.voxels.acquire(parent.children_index() + i as usize);
        }
        {
            let parent = self.nodes.acquire_mut(parent_index);
            parent.bitmask |= 0x1u64 << child_xyz;
            parent.data = parent.data & (0x1u32 << 31) | (new_child_array_index as u32);
        }
        (self.voxels.acquire_mut(new_child_index), new_child_index)
    }

    fn find_node(&self, pos: UVec3) -> (Option<&Node>, Vec<usize>) {
        let mut stack = Vec::with_capacity(self.tree_depth);
        let mut node = self.nodes.acquire(self.root_index);
        stack.push(self.root_index);
        while stack.len() < self.tree_depth {
            let child_width = 0x1u32 << (self.tree_depth - stack.len()) * 2;
            let child_local_pos = (pos & (child_width * 4 - 1)) / child_width;
            let child_xyz = child_local_pos.dot(UVec3::new(1, 4, 16));
            if (node.bitmask & (0x1u64 << child_xyz)) == 0 {
                return (None, stack);
            }
            let child_index = node.children_index() + (node.bitmask & !(u64::MAX << child_xyz)).count_ones() as usize;
            node = self.nodes.acquire(child_index);
            stack.push(child_index);
        }
        (Some(node), stack)
    }

    fn delete_node(&mut self, _node: &mut Node, _stack: &mut Vec<usize>) {
        unimplemented!()
    }
}

impl<T: Default + Copy + PartialEq + Display> Default for ConTree<T> {
    fn default() -> Self {
        let mut nodes = MemoryPool::default();
        let (_, root_index) = nodes.allocate();
        ConTree {
            nodes,
            voxels: MemoryPool::default(),
            root_index,
            tree_depth: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::UVec3;

    #[test]
    fn test_default_voxel_value() {
        let tree = ConTree::<u8>::default();
        assert_eq!(tree.get_voxel(UVec3::new(0, 0, 0)), u8::default());
        assert_eq!(tree.get_voxel(UVec3::new(10, 10, 10)), u8::default());
        assert_eq!(tree.get_voxel(UVec3::new(3, 3, 3)), u8::default());
    }

    #[test]
    fn test_set_and_get_voxel() {
        let mut tree = ConTree::<u8>::default();
        let pos = UVec3::new(1, 2, 3);

        tree.set_voxel(pos, 42);
        tree.set_voxel(pos, 43);
        assert_eq!(tree.get_voxel(pos), 43);

        // Other positions should still be default
        assert_eq!(tree.get_voxel(UVec3::new(0, 0, 0)), u8::default());
    }

    #[test]
    fn test_set_multiple_voxels() {
        let mut tree = ConTree::<u8>::default();

        tree.set_voxel(UVec3::new(1, 1, 1), 10);
        tree.set_voxel(UVec3::new(2, 2, 2), 20);
        tree.set_voxel(UVec3::new(3, 3, 3), 30);

        assert_eq!(tree.get_voxel(UVec3::new(1, 1, 1)), 10);
        assert_eq!(tree.get_voxel(UVec3::new(2, 2, 2)), 20);
        assert_eq!(tree.get_voxel(UVec3::new(3, 3, 3)), 30);

        // Undefined voxels should remain default
        assert_eq!(tree.get_voxel(UVec3::new(4, 4, 4)), u8::default());
    }

    #[test]
    fn test_chunk_boundary_voxel() {
        let mut tree = ConTree::<u8>::default();

        // Set a voxel at the boundary of a chunk (4x4x4)
        let pos = UVec3::new(3, 3, 3);
        tree.set_voxel(pos, 99);

        assert_eq!(tree.get_voxel(pos), 99);

        // The next voxel should still be default
        assert_eq!(tree.get_voxel(UVec3::new(4, 4, 4)), u8::default());
    }

    #[test]
    fn test_no_modulo_wrapping() {
        let mut tree = ConTree::<u8>::default();

        // Set voxel at (0, 0, 0)
        tree.set_voxel(UVec3::new(0, 0, 0), 100);

        // Check that (4,4,4) is still default and wasn't accidentally modified
        assert_eq!(tree.get_voxel(UVec3::new(4, 4, 4)), u8::default());

        // Now explicitly set (4,4,4) and check again
        tree.set_voxel(UVec3::new(4, 4, 4), 200);

        // Ensure both values are correct and independent
        assert_eq!(tree.get_voxel(UVec3::new(0, 0, 0)), 100);
        assert_eq!(tree.get_voxel(UVec3::new(4, 4, 4)), 200);
    }
}
