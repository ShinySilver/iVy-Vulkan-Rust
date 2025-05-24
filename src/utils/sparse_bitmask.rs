use glam::UVec3;

struct SparseBitmask {}

impl SparseBitmask {
    pub fn new(tree_depth: usize) -> Self { SparseBitmask {} }
    pub fn set(&mut self, pos: UVec3, value: bool) {}
    pub fn get(&self, pos: UVec3) -> bool { false }
    pub fn get_mut(&mut self, pos: UVec3) -> Option<&mut bool> { None }
}