use std::cmp::max;
use std::marker::PhantomData;

pub trait AbstractPredictor {}

pub struct NoPredictor {}

impl AbstractPredictor for NoPredictor {}

pub struct PackedMemoryArray<K, T, P>
where
    K: Ord + Clone,
    T: Clone,
    P: AbstractPredictor,
{
    capacity: usize,
    segment_capacity: usize,
    nb_segments: usize,
    nb_elements: usize,
    first_element_pos: usize,
    last_element_pos: usize,
    height: usize,
    t_h: f64,
    t_0: f64,
    p_h: f64,
    p_0: f64,
    t_d: f64,
    p_d: f64,
    array: Vec<Option<(K, T)>>,
    predictor: PhantomData<P>,
}

impl<K, T, P> PackedMemoryArray<K, T, P>
where
    K: Ord + Clone,
    T: Clone,
    P: AbstractPredictor,
{
    fn new(capacity: usize, t_h: f64, t_0: f64, p_h: f64, p_0: f64) -> Self {
        let nb_segments = (capacity as f64 / (capacity as f64).log2()).exp2().ceil() as usize;
        let segment_capacity = capacity / nb_segments;
        let height = (nb_segments as f64).log2().ceil() as usize;
        let t_d = (t_h - t_0) / height as f64;
        let p_d = (p_h - p_0) / height as f64;
        let array = vec![None; capacity];
        PackedMemoryArray {
            capacity,
            segment_capacity,
            nb_segments,
            nb_elements: 0,
            first_element_pos: 0,
            last_element_pos: 0,
            height,
            t_h,
            t_0,
            p_h,
            p_0,
            t_d,
            p_d,
            array,
            predictor: PhantomData,
        }
    }

    fn from_vecs(keys: Vec<K>, values: Vec<T>) -> Self {
        let capacity = keys.len();
        let mut pma = PackedMemoryArray::new(capacity, 0.7, 0.92, 0.3, 0.08);
        pma.nb_elements = keys.len();
        for (i, (k, v)) in keys.into_iter().zip(values.into_iter()).enumerate() {
            pma.array[i] = Some((k, v));
        }
        pma
    }

    fn nnz(&self) -> usize {
        self.nb_elements
    }

    fn iter(&self) -> impl Iterator<Item = &(K, T)> {
        self.array.iter().filter_map(|x| x.as_ref())
    }

    fn get(&self, key: &K) -> Option<&T> {
        self.array
            .iter()
            .find_map(|x| x.as_ref().and_then(|(k, v)| if k == key { Some(v) } else { None }))
    }

    fn contains_key(&self, key: &K) -> bool {
        self.array.iter().any(|x| x.as_ref().map_or(false, |(k, _)| k == key))
    }

    fn entry(&mut self, key: K) -> Entry<'_, K, T> {
        if let Some(pos) = self.array.iter().position(|x| x.as_ref().map_or(false, |(k, _)| k == key)) {
            Entry::Occupied(OccupiedEntry { pma: self, pos })
        } else {
            Entry::Vacant(VacantEntry { pma: self, key })
        }
    }

    fn even_rebalance(&mut self, window_start: usize, window_end: usize, m: usize) {
        let capacity = window_end - window_start + 1;
        if capacity == self.segment_capacity {
            return;
        }
        self.pack(window_start, window_end, m);
        self.spread(window_start, window_end, m);
    }

    fn look_for_rebalance(&mut self, pos: usize) -> (usize, usize, usize) {
        let mut p = 0.0;
        let mut t = 0.0;
        let mut density = 0.0;
        let mut height = 0;
        let mut prev_win_start = pos;
        let mut prev_win_end = pos - 1;
        let mut nb_cells_left = 0;
        let mut nb_cells_right = 0;
        while height <= self.height {
            let window_capacity = 1 << height * self.segment_capacity;
            let win_start = ((pos - 1) / window_capacity) * window_capacity + 1;
            let win_end = win_start + window_capacity - 1;
            nb_cells_left += self.nbcells(win_start, prev_win_start);
            nb_cells_right += self.nbcells(prev_win_end + 1, win_end + 1);
            density = (nb_cells_left + nb_cells_right) as f64 / window_capacity as f64;
            p = self.p_0 + self.p_d * height as f64;
            t = self.t_0 + self.t_d * height as f64;
            if p <= density && density <= t {
                let nb_cells = nb_cells_left + nb_cells_right;
                return (win_start, win_end, nb_cells);
            }
            prev_win_start = win_start;
            prev_win_end = win_end;
            height += 1;
        }
        let nb_cells = nb_cells_left + nb_cells_right;
        if density > t {
            self.extend();
        }
        if density < p && self.height > 1 {
            self.pack(1, self.capacity / 2, nb_cells);
            self.shrink();
        }
        (1, self.capacity, nb_cells)
    }

    fn extend(&mut self) {
        self.capacity *= 2;
        self.nb_segments *= 2;
        self.height += 1;
        self.t_d = (self.t_h - self.t_0) / self.height as f64;
        self.p_d = (self.p_h - self.p_0) / self.height as f64;
        self.array.resize(self.capacity, None);
    }

    fn shrink(&mut self) {
        self.capacity /= 2;
        self.nb_segments /= 2;
        self.height -= 1;
        self.t_d = (self.t_h - self.t_0) / self.height as f64;
        self.p_d = (self.p_h - self.p_0) / self.height as f64;
        self.array.truncate(self.capacity);
    }
    fn pack(&mut self, window_start: usize, window_end: usize, m: usize) {
        let mut j = window_start;
        for i in window_start..=window_end {
            if let Some((key, value)) = self.array[i] {
                self.array[j] = Some((key, value));
                j += 1;
            }
        }
        for i in j..=window_end {
            self.array[i] = None;
        }
    }

    fn spread(&mut self, window_start: usize, window_end: usize, m: usize) {
        let window_capacity = window_end - window_start + 1;
        let mut j = window_end;
        let mut k = m;
        let mut hole_size = window_capacity - m;
        let mut hole_capacity = hole_size;
        while k > 0 {
            if let Some((key, value)) = self.array[j] {
                let mut pos = ((hole_capacity - hole_size) as f64 / k as f64 * window_capacity as f64).floor() as usize + window_start;
                while pos < window_start || pos > window_end || self.array[pos].is_some() {
                    pos += 1;
                }
                self.array[pos] = Some((key, value));
                self.array[j] = None;
                k -= 1;
                hole_size -= 1;
            } else {
                hole_size -= 1;
            }
            j -= 1;
        }
    }

    fn nbcells(&self, start: usize, end: usize) -> usize {
        let mut count = 0;
        for i in start..end {
            if self.array[i].is_some() {
                count += 1;
            }
        }
        count
    }
}
}

pub enum Entry<'a, K, T>
where
    K: Ord + Clone,
    T: Clone,
{
    Occupied(OccupiedEntry<'a, K, T>),
    Vacant(VacantEntry<'a, K, T>),
}

pub struct OccupiedEntry<'a, K, T>
where
    K: Ord + Clone,
    T: Clone,
{
    pma: &'a mut PackedMemoryArray<K, T, NoPredictor>,
    pos: usize,
}

impl<'a, K, T> OccupiedEntry<'a, K, T>
where
    K: Ord + Clone,
    T: Clone,
{
    pub fn get(&self) -> &T {
        self.pma.array[self.pos].as_ref().unwrap().1
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.pma.array[self.pos].as_mut().unwrap().1
    }
}

pub struct VacantEntry<'a, K, T>
where
    K: Ord + Clone,
    T: Clone,
{
    pma: &'a mut PackedMemoryArray<K, T, NoPredictor>,
    key: K,
}

impl<'a, K, T> VacantEntry<'a, K, T>
where
    K: Ord + Clone,
    T: Clone,
{
    pub fn insert(self, value: T) -> &'a mut T {
        let pos = self.pma.array.iter().position(|x| x.is_none()).unwrap();
        self.pma.array[pos] = Some((self.key, value));
        self.pma.nb_elements += 1;
        self.pma.array[pos].as_mut().unwrap().1
    }
}
