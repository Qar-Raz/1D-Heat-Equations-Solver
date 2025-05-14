import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

//  Copyright (c) 2023 AUTHORS
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

public class HeatSimulation {

    static final int ghosts = 1;
    static int nx = 1000000;  // Hardcoded grid size
    static final double k = 0.4;
    static int nt = 10000;   // Hardcoded number of time steps
    static final double dt = 1.0;
    static final double dx = 1.0;
    static int threads = 4; // Hardcoded number of threads

    public static void pr(double[] total) {
        System.out.print("[");
        for (int i = 0; i < total.length; i++) {
            System.out.print(" " + total[i]);
        }
        System.out.println(" ]");
    }

    static class Queue {
        static final int QUEUE_CAPACITY = 20; // Renamed from sz to avoid confusion
        private final double[] data = new double[QUEUE_CAPACITY];
        private volatile int head = 0;
        private volatile int tail = 0;
        private final Object lock = new Object(); // Explicit lock object for wait/notify

        public void push(double d) {
            synchronized (lock) {
                while (tail - head >= QUEUE_CAPACITY) {
                    // This condition implies the queue is full.
                    // The C++ version yields, here we wait for space.
                    // System.err.println("Queue full, waiting. This should be rare.");
                    try {
                        lock.wait();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        System.err.println("Push interrupted.");
                        return;
                    }
                }
                data[tail % QUEUE_CAPACITY] = d;
                // The volatile write to tail ensures prior writes (like to data array)
                // are visible before tail is seen as updated by other threads.
                tail++;
                // Notify a single waiting pop() call.
                lock.notify();
            }
        }

        public double pop() {
            synchronized (lock) {
                while (head == tail) {
                    // Queue is empty, wait for an item.
                    try {
                        lock.wait();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        System.err.println("Pop interrupted.");
                        return Double.NaN; // Indicate error
                    }
                }
                double result = data[head % QUEUE_CAPACITY];
                // The volatile write to head ensures its update is visible.
                head++;
                // Notify a single waiting push() call (if it was waiting due to full queue).
                lock.notify();
                return result;
            }
        }
    }

    static class Worker extends Thread {
        int num;
        int lo, hi, sz;
        double[] data, data2;
        Queue left, right; // These are the queues THIS worker will POP from
        Worker leftThread = null;
        Worker rightThread = null;

        public Worker(int num, int tx) {
            super("Worker-" + num); // Name the thread for easier debugging
            this.num = num;

            lo = tx * num;
            hi = tx * (num + 1);
            if (hi > HeatSimulation.nx) {
                hi = HeatSimulation.nx;
            }
            lo -= HeatSimulation.ghosts;
            hi += HeatSimulation.ghosts;
            sz = hi - lo;

            data = new double[sz];
            data2 = new double[sz];

            // Queues this worker will receive from
            this.left = new Queue();
            this.right = new Queue();

            int off = 1;
            for (int n = 0; n < sz; n++) {
                data[n] = (double)n + lo + off; // Initial data
                data2[n] = 0.0;
            }
        }

        // No explicit destructor ~Worker() needed due to Java GC.

        void recv_ghosts() {
            // Pop from my left queue (data sent by leftThread to my left ghost cell)
            data[0] = left.pop();
            // Pop from my right queue (data sent by rightThread to my right ghost cell)
            data[sz - 1] = right.pop();
        }

        void update() {
            recv_ghosts();

            for (int n = 1; n < sz - 1; n++) {
                data2[n] = data[n] + HeatSimulation.k * HeatSimulation.dt / (HeatSimulation.dx * HeatSimulation.dx) *
                           (data[n + 1] + data[n - 1] - 2 * data[n]);
            }

            // Swap data and data2 arrays
            double[] temp = data;
            data = data2;
            data2 = temp;

            send_ghosts();
        }

        void send_ghosts() {
            // data[0] is left ghost cell, data[1] is the first "real" data point
            // data[sz-1] is right ghost cell, data[sz-2] is the last "real" data point

            // Send my first real data point (data[1]) to leftThread's right boundary queue
            leftThread.right.push(data[1]);
            // Send my last real data point (data[sz-2]) to rightThread's left boundary queue
            rightThread.left.push(data[sz - 2]);
        }

        @Override
        public void run() {
            send_ghosts(); // Initial exchange
            for (int t = 0; t < HeatSimulation.nt; t++) {
                update();
            }
            recv_ghosts(); // Final receive to make sure data array has updated ghost cells for construction
        }
    }

    public static double[] construct_grid(List<Worker> workers) {
        double[] total = new double[HeatSimulation.nx];
        for (Worker w : workers) {
            // w.lo is the global index of the start of w.data (which includes left ghost)
            // w.lo + ghosts is the global index of the first actual data point for this worker
            int current_global_idx = w.lo + HeatSimulation.ghosts;
            for (int n = HeatSimulation.ghosts; n < w.data.length - HeatSimulation.ghosts; n++) {
                if (current_global_idx >= 0 && current_global_idx < HeatSimulation.nx) { // Boundary check
                    total[current_global_idx] = w.data[n];
                }
                current_global_idx++;
            }
        }
        return total;
    }

    public static void main(String[] args) {
        // Values are hardcoded at the top of the class

        List<Worker> workerList = new ArrayList<>();
        // Calculate partition chunk size. Note: C++ tx calculation used.
        // This tx represents the span of grid points (excluding ghosts initially)
        // that each thread is notionally responsible for before ghost zones are added.
        int tx_param_for_worker_constructor = (2 * ghosts + nx) / threads;


        for (int th = 0; th < threads; th++) {
            workerList.add(new Worker(th, tx_param_for_worker_constructor));
        }

        for (int th = 0; th < threads; th++) {
            int next = (th + 1) % threads;
            int prev = (threads + th - 1) % threads;
            Worker w = workerList.get(th);
            w.rightThread = workerList.get(next);
            w.leftThread = workerList.get(prev);
        }

        long t1 = System.nanoTime();
        for (Worker w : workerList) {
            w.start(); // Calls Thread.start(), which executes Worker.run()
        }

        for (Worker w : workerList) {
            try {
                w.join();
            } catch (InterruptedException e) {
                System.err.println("Main thread interrupted while waiting for worker: " + w.getName());
                Thread.currentThread().interrupt(); // Preserve interrupt status
            }
        }
        long t2 = System.nanoTime();
        double elapsed = (t2 - t1) * 1e-9; // Convert nanoseconds to seconds
        System.out.println("elapsed: " + elapsed);

        double[] total = construct_grid(workerList);
        if (nx <= 20) {
            pr(total);
        }

        String perfDataFile = "perfdata.csv";
        boolean fileExists = Files.exists(Paths.get(perfDataFile));

        // Try-with-resources for automatic closing of FileWriter/PrintWriter
        try (FileWriter fw = new FileWriter(perfDataFile, true);
             BufferedWriter bw = new BufferedWriter(fw);
             PrintWriter f = new PrintWriter(bw)) {

            if (!fileExists) {
                f.println("lang,nx,nt,threads,dt,dx,total time,flops");
            }
            // Use Locale.US to ensure '.' as decimal separator, consistent with C++ output
            f.printf(Locale.US, "java,%d,%d,%d,%.1f,%.1f,%.9f,0%n",
                    nx, nt, threads, dt, dx, elapsed);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}