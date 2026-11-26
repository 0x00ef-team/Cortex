//! Tiny persistent thread pool used to parallelize matrix-vector products.
//!
//! Unlike `thread::scope`, the worker threads are created once when a model
//! is loaded. This avoids spawning thousands of threads per generated token.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Barrier};
use std::thread::{self, JoinHandle};

type JobFn = dyn Fn(usize) + Send + Sync;

struct Job {
    func: Arc<JobFn>,
    index: usize,
    barrier: Arc<Barrier>,
}

/// A fixed set of worker threads.
pub struct ThreadPool {
    senders: Vec<Sender<Job>>,
    _workers: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let mut senders = Vec::with_capacity(threads);
        let mut workers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let (tx, rx): (Sender<Job>, Receiver<Job>) = mpsc::channel();
            senders.push(tx);
            workers.push(thread::spawn(move || worker(rx)));
        }
        Self {
            senders,
            _workers: workers,
        }
    }

    pub fn len(&self) -> usize {
        self.senders.len()
    }

    /// Run `count` indexed jobs in parallel and wait for all of them.
    ///
    /// Jobs are distributed round-robin over the workers. The barrier ensures
    /// that borrowed data (passed as raw pointers by the caller) stays valid
    /// until every job has finished.
    pub fn dispatch<F>(&self, count: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        if count == 0 {
            return;
        }
        let func: Arc<JobFn> = Arc::new(f);
        let barrier = Arc::new(Barrier::new(count + 1));
        for i in 0..count {
            let job = Job {
                func: Arc::clone(&func),
                index: i,
                barrier: Arc::clone(&barrier),
            };
            self.senders[i % self.senders.len()]
                .send(job)
                .expect("thread pool worker has stopped");
        }
        barrier.wait();
    }
}

fn worker(rx: Receiver<Job>) {
    while let Ok(job) = rx.recv() {
        (job.func)(job.index);
        job.barrier.wait();
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self::new(n)
    }
}
