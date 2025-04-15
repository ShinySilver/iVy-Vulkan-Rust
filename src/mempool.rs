pub struct MemoryPool<T: Default + Copy> {
    memory: Vec<T>,
    holes: [Vec<usize>; 64],
}

impl<T: Default + Copy> MemoryPool<T> {
    pub fn new(size: usize) -> Self {
        Self {
            memory: Vec::with_capacity(size),
            holes: core::array::from_fn(|_| Vec::new()),
        }
    }

    pub fn allocate(&mut self) -> (&mut T, usize) {
        match self.holes[0].pop() {
            Some(index) => (&mut self.memory[index], index),
            None => {
                assert!(self.memory.len() + 1 <= self.memory.capacity(), "Out of memory!");
                let index = self.memory.len();
                self.memory.push(T::default());
                (&mut self.memory[index], index)
            }
        }
    }

    pub fn allocate_multiple(&mut self, count: usize) -> (&mut [T], usize) {
        assert!(count > 0 && count <= 64, "N must be between 1 and 64");
        match self.holes[count - 1].pop() {
            Some(index) => (&mut self.memory[index..index + count], index),
            None => {
                assert!(self.memory.len() + count <= self.memory.capacity(), "Out of memory!");
                let index = self.memory.len();
                self.memory.extend(std::iter::repeat(T::default()).take(count));
                (&mut self.memory[index..], index)
            }
        }
    }

    pub fn deallocate(&mut self, index: usize) {
        self.holes[0].push(index);
        self.memory[index] = T::default();
    }

    pub fn deallocate_multiple(&mut self, index: usize, count: usize) {
        assert!(count > 0 && count <= 64, "N must be between 1 and 64!");
        self.holes[count - 1].push(index);
        self.memory[index..index + count].fill(Default::default());
    }

    pub fn acquire(&self, index: usize) -> &T {
        assert!(index < self.memory.len(), "Invalid index!");
        &self.memory[index]
    }

    pub fn acquire_mut(&mut self, index: usize) -> &mut T {
        assert!(index < self.memory.len(), "Invalid index!");
        &mut self.memory[index]
    }

    pub fn acquire_multiple(&self, index: usize, count: usize) -> &[T] {
        assert!(count > 0 && count <= 64, "N must be between 1 and 64!");
        assert!(index + count <= self.memory.len(), "Invalid index!");
        &self.memory[index..index + count]
    }

    pub fn acquire_multiple_mut(&mut self, index: usize, count: usize) -> &mut [T] {
        assert!(count > 0 && count <= 64, "N must be between 1 and 64!");
        assert!(index + count <= self.memory.len(), "Invalid index!");
        &mut self.memory[index..index + count]
    }
}

impl<T: Default + Copy> Default for MemoryPool<T> {
    fn default() -> Self {
        Self {
            memory: Vec::with_capacity(4096),
            holes: core::array::from_fn(|_| Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test with i32 type
    #[test]
    fn test_allocate_i32() {
        let mut pool = MemoryPool::<i32>::new(10);
        let (_, index) = pool.allocate();
        assert_eq!(index, 0);

        let (_, index) = pool.allocate();
        assert_eq!(index, 1);
    }

    // Test default constructor
    #[test]
    fn test_default_trait() {
        let mut pool = MemoryPool::<i32>::default();
        let (_, index) = pool.allocate();
        assert_eq!(index, 0);

        let (_, index) = pool.allocate();
        assert_eq!(index, 1);
    }

    // Test with i32 type for allocating multiple
    #[test]
    fn test_allocate_multiple_i32() {
        let mut pool = MemoryPool::<i32>::new(10);
        let (_, index) = pool.allocate_multiple(3);
        assert_eq!(index, 0);

        let (_, index) = pool.allocate_multiple(2);
        assert_eq!(index, 3);
    }

    // Test deallocation with i32 type
    #[test]
    fn test_deallocate_i32() {
        let mut pool = MemoryPool::<i32>::new(10);
        let (_, index) = pool.allocate();
        pool.deallocate(index);
        let (_, new_index) = pool.allocate();
        assert_eq!(index, new_index);
    }

    // Test multiple deallocations with i32 type
    #[test]
    fn test_deallocate_multiple_i32() {
        let mut pool = MemoryPool::<i32>::new(10);
        let (_, index) = pool.allocate_multiple(3);
        pool.deallocate_multiple(index, 3);
        let (_, new_index) = pool.allocate_multiple(3);
        assert_eq!(index, new_index);
    }

    // Test for out of memory panic with i32 type
    #[test]
    #[should_panic(expected = "Out of memory!")]
    fn test_out_of_memory_i32() {
        let mut pool = MemoryPool::<i32>::new(3);
        pool.allocate();
        pool.allocate();
        pool.allocate();
        pool.allocate(); // Should panic
    }

    // Test invalid acquire with i32 type
    #[test]
    #[should_panic(expected = "Invalid index!")]
    fn test_invalid_acquire_i32() {
        let pool = MemoryPool::<i32>::new(3);
        pool.acquire(5); // Should panic
    }

    // Test invalid allocate_multiple with i32 type
    #[test]
    #[should_panic(expected = "N must be between 1 and 64")]
    fn test_invalid_allocate_multiple_i32() {
        let mut pool = MemoryPool::<i32>::new(10);
        pool.allocate_multiple(65); // Should panic
    }
}