//! An insertion-ordered map with the semantics of a Python `dict`: a new
//! key goes at the end, updating a key keeps its place, a removed key
//! inserted again goes at the end. The ports use it where the order of a
//! Python dict reaches the results.

use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, Clone)]
pub struct OrderedMap<K: Hash + Eq + Clone, V> {
    slots: Vec<Option<(K, V)>>,
    index: HashMap<K, usize>,
}

impl<K: Hash + Eq + Clone, V> Default for OrderedMap<K, V> {
    fn default() -> Self {
        OrderedMap::new()
    }
}

impl<K: Hash + Eq + Clone, V: PartialEq> PartialEq for OrderedMap<K, V> {
    /// Python's dict equality: the same keys and values, any order.
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}

impl<K: Hash + Eq + Clone, V> OrderedMap<K, V> {
    pub fn new() -> Self {
        OrderedMap {
            slots: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.index
            .get(key)
            .map(|&i| &self.slots[i].as_ref().expect("a live slot").1)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// `d[key] = value`.
    pub fn insert(&mut self, key: K, value: V) {
        match self.index.get(&key) {
            Some(&i) => self.slots[i].as_mut().expect("a live slot").1 = value,
            None => {
                self.index.insert(key.clone(), self.slots.len());
                self.slots.push(Some((key, value)));
            }
        }
    }

    /// `d.pop(key, None)`.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let i: usize = self.index.remove(key)?;
        let (_, value) = self.slots[i].take().expect("a live slot");
        if self.slots.len() > 32 && self.index.len() * 2 < self.slots.len() {
            self.compact();
        }
        Some(value)
    }

    fn compact(&mut self) {
        let slots: Vec<Option<(K, V)>> = std::mem::take(&mut self.slots);
        self.index.clear();
        for (k, v) in slots.into_iter().flatten() {
            self.index.insert(k.clone(), self.slots.len());
            self.slots.push(Some((k, v)));
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.slots.iter().filter_map(|s| s.as_ref().map(|(k, v)| (k, v)))
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }
}

impl<K: Hash + Eq + Clone, V> FromIterator<(K, V)> for OrderedMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(items: I) -> Self {
        let mut map: OrderedMap<K, V> = OrderedMap::new();
        for (k, v) in items {
            map.insert(k, v);
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_dict_order() {
        let mut d: OrderedMap<i32, i32> = OrderedMap::new();
        d.insert(1, 10);
        d.insert(2, 20);
        d.insert(1, 11);
        d.remove(&2);
        d.insert(3, 30);
        d.insert(2, 21);
        assert_eq!(
            d.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(),
            vec![(1, 11), (3, 30), (2, 21)]
        );
    }
}
