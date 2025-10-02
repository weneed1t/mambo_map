A very simple and cross-platform implementation of the cache table framework that uses only use std::{
mem,
    sync::{Arc, Mutex, RwLock, TryLockError},
} set of the standard rust library

```rust

    #[test]
    fn based_example() {
        {
            let shards = 16;
            let elems_in_mutex = 7.0;
            const NUM_THREADS: usize = 10;
            const OPS_PER_THREAD: usize = 100;
            let mambo = Mambo::<String>::new(shards, elems_in_mutex).unwrap();

            for tt in 1..NUM_THREADS {
                let mut mambo_arc = mambo.arc_clone();

                let _ = thread::spawn(move || {
                    for key in 0..OPS_PER_THREAD {
                        let key = tt + (key * OPS_PER_THREAD * 10);

                        let elem_me = format!("mambo elem{}", key);

                        mambo_arc.insert(key as u64, &elem_me, false).unwrap();

                        assert_eq!(mambo_arc.insert(key as u64, &elem_me, false), Ok(None));

                        mambo_arc
                            .read(key as u64, |ind| {
                                let ind = ind.unwrap();

                                assert_eq!(
                                    ind.clone(),
                                    elem_me,
                                    " non eq read  key: {}   rea: {}   in map: {}",
                                    key,
                                    elem_me,
                                    ind.clone()
                                );
                                Ok(())
                            })
                            .unwrap();
                    }

                    for key in 0..OPS_PER_THREAD {
                        let key = tt + (key * OPS_PER_THREAD * 10);
                        let elem_me = format!("mambo elem{}", key);

                        assert_eq!(mambo_arc.remove(key as u64), Ok(Some(elem_me.clone())));
                    }
                });
            }
        }
    }
```


```rust
 pub fn new(num_shards: usize, redream_factor: f32) -> Result<Self, &'static str>
 ```
The num_shards table is divided into independent sections for optimization,
the optimal value is num_shards = the number of cores on your processor for multithreaded operation.
The redream_factor can be from 1.0 to 10.0, depending on how Mambo
is planned to be applied. ->10.0 when a lot of elements are planned (more than 10_000 ).
if there are few elements, about 1000 or less, it is better to use values tending to ->1.0.
At ->10, memory consumption per element decreases (depending on the system, at 1.0,
memory costs per element range from 32 to 64 bytes of overhead memory,
at ->10.0 per 1 element costs are reduced to ~10 bytes per element)


```rust
pub fn arc_clone(&self) -> Self
 ```
Some of the Mambo structure data is thread-independent,
and each copy via arc_clone provides a convenient instance of the Mambo structure that can be used
in a new thread. all data regarding the elements of the Mambo hash table can be obtained from any
stream that has an instance of arc_clone.


```rust
 pub fn read<RFy>(&mut self, key: u64, ridler: RFy) -> Result<(), &'static str> 
 ```
Attention!! DO NOT CALL read INSIDE THE read CLOSURE!!! THIS MAY CAUSE THE THREAD TO SELF-LOCK!!

reading. to access an item for reading, you need to request it using the key.
and the element is read and processed in the
RFy closure:FnOnce(Option<&mut T>) -> Result<(), &'static str>,
since &mut T is Mutex.lock(). while processing is taking place in read< RFy >,
the shard in which this element is located cannot:
1 change its size.
2: since Mutex contains elements from 0 to redream_factor
(a parameter in the Mamdo::new constructor),
access in other threads for elements in one Mutex is blocked.
to summarize, the faster the closure is resolved inside read, the better.


```rust
pub fn remove(&mut self, key: u64) -> Result<Option<T>, &'static str> 
 ```
to remove the fragment. you need to return the key key: u64
if an element was in the table and it was deleted, but the function returns Ok(Some(T.clone()))
if there was no such element in the hash table, Ok(None) will be returned


```rust
pub fn insert(&mut self,key: u64,elem: &T,force_replace: bool) -> Result<Option<T>, &'static str>
 ```
insert an element T with the key: u64.if there is already an element with the same key: u64 in the table,
then when force_replace == true, the old element T will be replaced by the new element T,
while the old element T will be returned as Ok(Some(T.clone())). if force_replace == false,
the old element will not be replaced and the function will return Ok(None).
if there is no element with this key: u64,
a new element will be added to the table and the function will output Ok(None)
