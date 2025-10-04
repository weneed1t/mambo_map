A very simple and cross-platform implementation of the hashtable that uses only use std::{
mem,
    sync::{Arc, Mutex, RwLock, TryLockError},
} set of the standard rust library

```rust
   #[test]
    fn based_example() {
        const NUM_THREADS: usize = 10;
        const OPS_PER_THREAD: usize = 10;
        let mut std_handles = Vec::new();
        let shift = 1_000_000u64;
        let global_counter = Arc::new(Mutex::new(0usize));
        //=======================================================================
        //The number of shards, a shard is an independent piece of the hash table,
        //the optimal number of shards = the number of threads.
        let shards = 16;
        /*elements that are added to the hash table
        A hash table has a data storage topology
        Arc<(f32, Box<[(Mutex<u32>, RwLock<(Mutex<usize>, [Box<[Mutex<(bool, Vec<(T, u64)>)>]>; 2])>)]>)>.
        Each individual Mutex can have an average of 1.0 to 10.0 elements.
        the more elements there are in the Mutex, the lower the overhead of storing in memory per element,
        but the higher the chance that one thread will block another when reading one element.*/
        let elems_in_mutex = 7.0;
        //returns Err(&str) if the data is incorrect
        let mut mambo = Mambo::<String>::new(shards, elems_in_mutex).unwrap();

        for tt in 1..NUM_THREADS {
            let mut mambo_arc = mambo.arc_clone();
            let arc_global_counter = Arc::clone(&global_counter);
            std_handles.push(thread::spawn(move || {
                for key in 0..OPS_PER_THREAD {
                    let key = (tt + (key * OPS_PER_THREAD * 10)) + shift as usize;

                    let elem_me = format!("mambo elem {}", key);

                    /*to insert an element, if an element with the same key value already exists,
                    it will return Some(T.clone())
                    false indicates whether it is necessary to forcibly replace the element with a new one,
                     even if there is already an element with such a key*/
                    assert_eq!(mambo_arc.insert(key as u64, &elem_me, false), None);

                    assert_eq!(
                        mambo_arc.insert(key as u64, &elem_me, false),
                        Some(elem_me.clone())
                    );
                    /*reading, because the Mutex is kept open during reading,
                    as long as there is an active read operation in the shard,
                    a resizing operation and a filter operation cannot be applied to the shard.*/
                    //NEVER CALL A RECURSIVE READ INSIDE A read() CLOSURE, AS THIS MAY LEAD TO MUTUAL LOCKING!!
                    mambo_arc.read(key as u64, |ind| {
                        let ind = ind.unwrap();

                        assert_eq!(
                            ind.clone(),
                            elem_me,
                            " non eq read  key: {}   rea: {}   in map: {}",
                            key,
                            elem_me,
                            ind.clone()
                        );
                    });

                    let key_to_filter = {
                        let mut mutexer = arc_global_counter.lock().unwrap();
                        let key_to_filter = *mutexer;
                        *mutexer += 1;
                        key_to_filter
                    };
                    let elem_me = format!(
                        "elem {} {} theread: {}{}",
                        key_to_filter,
                        if key_to_filter > 9 { "" } else { " " },
                        tt,
                        if tt > 9 { "" } else { " " },
                    );
                    assert_eq!(
                        mambo_arc.insert(key_to_filter as u64, &elem_me, false),
                        None
                    );
                }

                for key in 0..OPS_PER_THREAD {
                    let key = tt + (key * OPS_PER_THREAD * 10) + shift as usize;
                    let elem_me = format!("mambo elem {}", key);
                    //deleting an element, if such an element exists,
                    // it will be deleted from the table and returned as (T.clone())
                    assert_eq!(mambo_arc.remove(key as u64), Some(elem_me.clone()));
                }
            }));
        }

        for handle in std_handles {
            handle.join().unwrap();
        }

        println!("before filter:");
        mambo.filter(|mut_elem, key_u64| {
            /*The filter option is needed when you need to filter out all the elements in the hashtable
            or change them, passes a closure with the desired parameters RFy: FnMut(&mut T, u64) -> bool,
            where &mut T is an element and u64 is its key.bool is a decision to delete an item or not,
            if bool == true, then this item will not be deleted from the cache table. if bool ==  false,
            then the element will be deleted*/
            println!(
                "    elem: {}, {} key: {}",
                *mut_elem,
                if key_u64 > 9 { "" } else { " " },
                key_u64
            );

            *mut_elem += " is even";
            /*an example that leaves in the table only those elements that are divisible by 2 without remainder*/
            if 0 == key_u64 % 2 {
                return true;
            }
            false
        });

        println!("after filter: ");
        mambo.filter(|mut_elem, key_u64| {
            println!(
                "    elem: {:<28}, {} key: {}",
                *mut_elem,
                if key_u64 > 9 { "" } else { " " },
                key_u64
            );

            true
        });
    }
```


```
Panic and Mutex handling:
Since Mutex and Rwlock have a Poison state, 
this is the state when the thread that was holding Guard caused a Panic and was destroyed.
It was decided that the user would determine for 
himself how to react to a panic the destruction of data in Mambo is implemented as follows:

1. if a panic was caused during the execution of mambo.read(),
the Mutex with the elements will be destroyed and replaced with an empty one, 
the data of this Mutex will be destroyed.

2. If a Panic occurred during the execution of 
mambo.filter() or a private method for resizing the shard,
 the shard will be destroyed and the data from it will be deleted 

3. Some errors in the thread can be fatal and will
cause irreversible data corruption in Mambo,
since Mambo uses only secure primitives,
a panic in case of such damage
will be caused in other threads that will access fatally corrupted data.
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
thread that has an instance of arc_clone.


```rust
 pub fn read<RFy>(&mut self, key: u64, ridler: RFy) 
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
pub fn remove(&mut self, key: u64) -> Option<T>
 ```
to remove the fragment. you need to return the key key: u64
if an element was in the table and it was deleted, but the function returns Some(T.clone())
if there was no such element in the hash table, None will be returned


```rust
pub fn insert(&mut self,key: u64,elem: &T,force_replace: bool) -> Option<T>
 ```
insert an element T with the key: u64.if there is already an element with the same key: u64 in the table,
then when force_replace == true, the old element T will be replaced by the new element T,
while the old element T will be returned as Some(T.clone()). if force_replace == false,
the old element will not be replaced and the function will return None.
if there is no element with this key: u64,
a new element will be added to the table and the function will output None


```rust
pub fn elems_im_me(&self) -> usize
 ```
returns the number of all items in the table.note.
for optimization in multithreaded environments,

!!!item count! changes do NOT occur EVERY TIME an item is DELETED/INSERTED.!!!
with a large number of elements and in a multithreaded environment,
it may not be critical, but when there are few elements, when elems_im_me is called,
it may return 0 even if there are elements in Mambo.
This is a forced decision to increase productivity.

## performance test in the --release version on the i7-11700f processor

|  threads    |  type   |millions op/ sec|
| ----------- |:-------:| -------------- |
| threads: 1  |  read   | M.op/S: 42.353 |
| threads: 2  |  read   | M.op/S: 23.663 |
| threads: 3  |  read   | M.op/S: 24.516 |
| threads: 4  |  read   | M.op/S: 27.592 |
| threads: 5  |  read   | M.op/S: 28.519 |
| threads: 6  |  read   | M.op/S: 29.785 |
| threads: 7  |  read   | M.op/S: 30.854 |
| threads: 8  |  read   | M.op/S: 31.136 |
| threads: 9  |  read   | M.op/S: 37.693 |
| threads: 10 |  read   | M.op/S: 42.285 |
| threads: 11 |  read   | M.op/S: 42.921 |
| threads: 12 |  read   | M.op/S: 46.473 |
| threads: 13 |  read   | M.op/S: 48.670 |
| threads: 14 |  read   | M.op/S: 49.510 |
| threads: 15 |  read   | M.op/S: 49.499 |
| threads: 16 |  read   | M.op/S: 52.769 |
| threads: 17 |  read   | M.op/S: 53.079 |
| threads: 18 |  read   | M.op/S: 54.096 |
| threads: 19 |  read   | M.op/S: 52.944 |
