use crate::utils::memory_pool::MemoryPool;
use glam::{uvec3, UVec3};

#[derive(Copy, Clone)]
struct Node {
    child_uniform: u8,  // Bit = 1 → child is uniform
    child_type: u8,     // Bit value when child is uniform
    children_index: u32, // Index into memory pool (non-uniform only)
}

impl Default for Node {
    fn default() -> Self {
        Node {
            child_uniform: u8::MAX,
            child_type: 0u8,
            children_index: 0u32,
        }
    }
}

#[derive(Default)]
pub struct SparseBitmask {
    tree_depth: usize,
    root_index: usize,
    nodes: MemoryPool<Node>,
    cached_pos: UVec3,
    cached_width: u32,
    cached_index: usize,
}


impl SparseBitmask {
    pub fn new(tree_depth: usize) -> Self {
        let mut nodes = MemoryPool::<Node>::new(16 * 1024 * 1024);
        let (_, root_index) = nodes.allocate();
        Self {
            tree_depth,
            root_index,
            nodes,
            ..Default::default()
        }
    }

    pub fn set(&mut self, pos: UVec3, value: bool) {
        // basically, lookup
        let (is_set, _, mut stack) = self.lookup(pos);

        // if you hit the right value, early exit
        if is_set == value {
            return;
        }

        // while you've not reached the max depth, create child that is mostly uniform, aside from child_xyz
        while stack.len() < self.tree_depth {
            // create a children array for the parent with incremented size
            let parent = *self.nodes.acquire(*stack.last().unwrap());
            let parent_child_array_size = (!parent.child_uniform).count_ones() as usize;
            let (_, new_child_array_index) = self.nodes.allocate_multiple(parent_child_array_size + 1);

            // now calculating the new child index in the new children array
            let child_width = 0x1u32 << (self.tree_depth - stack.len());
            let child_local_pos = (pos & (child_width * 2 - 1)) / child_width; // TODO: improve: check MSB of a bitmask
            let child_xyz = child_local_pos.dot(uvec3(1, 2, 4));
            assert!(child_xyz < 8);
            let child_index = (!parent.child_uniform & !(u8::MAX << child_xyz)).count_ones();

            // copy the original children array to the new children array, with a hole in the middle for the new child
            let new_child_index = new_child_array_index + child_index as usize;
            for i in 0..child_index as usize {
                *self.nodes.acquire_mut(new_child_array_index + i) =
                    *self.nodes.acquire(parent.children_index as usize + i);
            }
            for i in child_index as usize..parent_child_array_size {
                *self
                    .nodes.acquire_mut(new_child_array_index + i + 1) =
                    *self.nodes.acquire(parent.children_index as usize + i);
            }

            // deleting the previous children array
            if parent.children_index != 0 { self.nodes.deallocate_multiple(parent.children_index as usize, parent_child_array_size) }


            // Modifying the parent
            {
                let parent = self.nodes.acquire_mut(*stack.last().unwrap());
                parent.children_index = new_child_array_index as u32;
                parent.child_uniform &= !(0x1u8 << child_xyz);
            }

            // initialize the new child with fully uniform value "is_set"...
            let new_child = self.nodes.acquire_mut(new_child_index);
            *new_child = Node {
                child_uniform: u8::MAX,
                child_type: if is_set { u8::MAX } else { 0u8 },
                children_index: 0,
            };
            stack.push(new_child_index);
        }

        // set the value in the leaf
        {
            let leaf = self.nodes.acquire_mut(*stack.last().unwrap());
            let child_width = 0x1u32 << (self.tree_depth - stack.len());
            let child_local_pos = (pos & (child_width * 2 - 1)) / child_width; // TODO: improve: check MSB of a bitmask
            let child_xyz = child_local_pos.dot(uvec3(1, 2, 4));
            assert!(child_xyz < 8);
            if value {
                leaf.child_type |= 1 << child_xyz;
            } else {
                leaf.child_type &= !(1 << child_xyz);
            }
        }

        // while the current node is uniform, delete and go up
        for i in (1..stack.len()).rev() {
            let child_index = stack[i];
            let parent_index = stack[i - 1];

            // Check if the child is uniform
            let is_uniform = {
                let child = self.nodes.acquire(child_index);
                child.child_uniform == u8::MAX && (child.child_type == 0u8 || child.child_type == u8::MAX)
            };

            if !is_uniform { break; }

            // Get child position in parent's children array
            let child_width = 0x1u32 << (self.tree_depth - i);
            let child_local_pos = (pos & (child_width * 2 - 1)) / child_width;
            let child_xyz = child_local_pos.dot(uvec3(1, 2, 4));
            assert!(child_xyz < 8);

            // Fully copy the parent node, skipping the deleted child
            let parent = *self.nodes.acquire(parent_index);
            let parent_child_array_size = (!parent.child_uniform).count_ones() as usize;
            let child_offset = (!parent.child_uniform & !(u8::MAX << child_xyz)).count_ones() as usize;
            let new_size = parent_child_array_size - 1;
            let (_, new_children_index) = self.nodes.allocate_multiple(new_size);
            for j in 0..child_offset {
                let value = *self.nodes.acquire(parent.children_index as usize + j);
                *self.nodes.acquire_mut(new_children_index + j) = value;
            }
            for j in child_offset + 1..parent_child_array_size {
                let value = *self.nodes.acquire(parent.children_index as usize + j);
                *self.nodes.acquire_mut(new_children_index + j - 1) = value;
            }

            // Free old children array
            if parent.children_index != 0 {
                self.nodes.deallocate_multiple(parent.children_index as usize, parent_child_array_size);
            }

            // Now safely mutate the parent
            let parent_mut = self.nodes.acquire_mut(parent_index);
            parent_mut.children_index = new_children_index as u32;
            parent_mut.child_uniform |= 1 << child_xyz;
            parent_mut.child_type &= !(1 << child_xyz);
            if value {
                parent_mut.child_type |= 1 << child_xyz;
            }
        }
    }

    pub fn get(&self, pos: UVec3) -> bool { self.lookup(pos).0 }

    pub fn size(&self) -> usize { self.nodes.size() }

    pub(crate) fn fast_unsafe_lookup(&mut self, pos: UVec3) -> bool {
        let origin = self.cached_pos;
        let width = self.cached_width;
        if (pos.x >= origin.x && pos.x < origin.x + width) &&
            (pos.y >= origin.y && pos.y < origin.y + width) &&
            (pos.z >= origin.z && pos.z < origin.z + width)
        {
            let node = self.nodes.acquire(self.cached_index);
            let child_local_pos = (pos & (self.cached_width * 2 - 1)) / self.cached_width;
            let child_xyz = child_local_pos.dot(uvec3(1, 2, 4));
            let _is_uniform = node.child_uniform & (0x1u8 << child_xyz) != 0;
            let is_set = node.child_type & (0x1u8 << child_xyz) != 0;
            return is_set;
        }
        let mut stack = Vec::with_capacity(self.tree_depth);
        let mut node = self.nodes.acquire(self.root_index);
        stack.push(self.root_index);
        while stack.len() <= self.tree_depth {
            let child_width = 0x1u32 << (self.tree_depth - stack.len());
            let child_local_pos = (pos & (child_width * 2 - 1)) / child_width; // TODO: improve: check MSB of a bitmask
            let child_xyz = child_local_pos.dot(uvec3(1, 2, 4));
            let is_uniform = node.child_uniform & (0x1u8 << child_xyz) != 0;
            let is_set = node.child_type & (0x1u8 << child_xyz) != 0;
            if is_uniform {
                self.cached_index = stack.pop().unwrap();
                self.cached_width = child_width;
                self.cached_pos = pos & !(child_width - 1);
                return is_set;
            }
            let child_index = node.children_index
                + (!node.child_uniform & !(u8::MAX << child_xyz)).count_ones();
            node = self.nodes.acquire(child_index as usize);
            stack.push(child_index as usize);
        }
        panic!("Unexpected tree exit!");
    }

    fn lookup(&self, pos: UVec3) -> (bool, &Node, Vec<usize>) {
        let mut stack = Vec::with_capacity(self.tree_depth);
        let mut node = self.nodes.acquire(self.root_index);
        stack.push(self.root_index);
        while stack.len() <= self.tree_depth {
            let child_width = 0x1u32 << (self.tree_depth - stack.len());
            let child_local_pos = (pos & (child_width * 2 - 1)) / child_width; // TODO: improve: check MSB of a bitmask
            let child_xyz = child_local_pos.dot(uvec3(1, 2, 4));
            let is_uniform = node.child_uniform & (0x1u8 << child_xyz) != 0;
            let is_set = node.child_type & (0x1u8 << child_xyz) != 0;
            if is_uniform {
                return (is_set, node, stack);
            }
            let child_index = node.children_index
                + (!node.child_uniform & !(u8::MAX << child_xyz)).count_ones();
            node = self.nodes.acquire(child_index as usize);
            stack.push(child_index as usize);
        }
        panic!("Unexpected tree exit!");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::uvec3;

    #[test]
    fn test_default_is_false() {
        let mut bitmask = SparseBitmask::new(4);
        assert!(!bitmask.get(uvec3(0, 0, 0)));
        assert!(!bitmask.get(uvec3(7, 7, 7)));
        assert!(!bitmask.get(uvec3(15, 15, 15))); // max in 4-level tree
    }

    #[test]
    fn test_set_and_get_true() {
        let mut bitmask = SparseBitmask::new(4);
        let pos = uvec3(3, 5, 7);
        bitmask.set(pos, true);
        assert!(bitmask.get(pos));
    }

    #[test]
    fn test_set_memory_usage() {
        let mut bitmask = SparseBitmask::new(4);
        let pos = uvec3(1, 2, 3);
        let size_before = bitmask.nodes.size();
        bitmask.set(pos, true);
        bitmask.set(pos, false);
        let size_after = bitmask.nodes.size();
        assert_eq!(size_before, size_after, "Memory usage should be unchanged after setting then resetting a value");
    }

    #[test]
    fn test_set_with_fuzzing() {
        use rand::Rng;
        let depth = 4; // 128
        let size = 1 << (depth - 1);
        let mut rng = rand::rng();
        let mut bitmask = SparseBitmask::new(depth);
        let mut reference = vec![vec![vec![false; size]; size]; size];
        for _ in 0..1_000 {
            let x = rng.random_range(0..size);
            let y = rng.random_range(0..size);
            let z = rng.random_range(0..size);
            let value = rng.random_bool(0.5);
            let pos = uvec3(x as u32, y as u32, z as u32);
            bitmask.set(pos, value);
            reference[x][y][z] = value;
            for x in 0..size {
                for y in 0..size {
                    for z in 0..size {
                        let pos = uvec3(x as u32, y as u32, z as u32);
                        let expected = reference[x][y][z];
                        let actual = bitmask.get(pos);
                        assert_eq!(actual, expected, "Mismatch at ({}, {}, {})", x, y, z);
                    }
                }
            }
        }
    }
}
