//  Copyright (c) 2023 AUTHORS
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

const std = @import("std");
const Thread = std.Thread;
const Mutex = Thread.Mutex;
const Condition = Thread.Condition;
const ArrayList = std.ArrayList;
const fs = std.fs;
const time = std.time;
const Allocator = std.mem.Allocator;
const print = std.debug.print; // For convenience

// Global constants and variables (matching C++)
const ghosts: usize = 1;
var nx: usize = 50000; // Hardcoded grid size
const k: f64 = 0.4;
var nt: usize = 100; // Hardcoded number of time steps
const dt: f64 = 1.0;
const dx: f64 = 1.0;
var threads_count: usize = 4; // Hardcoded number of threads (renamed from 'threads' to avoid conflict with std.Thread)

fn pr_grid(total: []const f64) !void {
    const stdout_writer = std.io.getStdOut().writer();
    try stdout_writer.print("[", .{});
    for (total) |val| {
        try stdout_writer.print(" {d}", .{val});
    }
    try stdout_writer.print(" ]\n", .{});
}

const Queue = struct {
    const SZ: usize = 20;
    data: [SZ]f64,
    head: usize,
    tail: usize,
    mutex: Mutex,
    cv: Condition,

    pub fn init() Queue {
        return Queue{
            .data = undefined,
            .head = 0,
            .tail = 0,
            .mutex = .{}, // Changed
            .cv = .{}, // Changed
        };
    }

    pub fn push(self: *Queue, d: f64) void {
        while (self.tail -% self.head >= SZ) {
            std.Thread.yield();
        }
        const current_tail = self.tail;
        self.data[current_tail % SZ] = d;
        @fence(.SeqCst); // Changed
        self.tail = current_tail +% 1;
        @fence(.SeqCst); // Changed

        if (self.head +% 1 == self.tail) {
            self.mutex.lock();
            self.cv.signal();
            self.mutex.unlock();
        }
    }

    pub fn pop(self: *Queue) f64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (self.head == self.tail) {
            self.cv.wait(&self.mutex);
        }

        const current_head = self.head;
        const result = self.data[current_head % SZ];
        @fence(.SeqCst); // Changed
        self.head = current_head +% 1;
        return result;
    }
};
const Worker = struct {
    num: usize,
    lo: usize,
    hi: usize,
    sz: usize,
    data: ArrayList(f64),
    data2: ArrayList(f64),
    left_q: Queue,
    right_q: Queue,
    leftThread: ?*Worker = null,
    rightThread: ?*Worker = null,
    thread_handle: ?Thread = null,
    allocator: Allocator, // Added for thread spawning

    pub fn init(allocator_param: Allocator, num_in: usize, tx_param: usize) !Worker {
        var self = Worker{
            .num = num_in,
            .lo = 0,
            .hi = 0,
            .sz = 0,
            .data = ArrayList(f64).init(allocator_param),
            .data2 = ArrayList(f64).init(allocator_param),
            .left_q = Queue.init(),
            .right_q = Queue.init(),
            .allocator = allocator_param, // Store the allocator
        };

        const data_region_lo: usize = tx_param * self.num; // Changed to const
        var data_region_hi: usize = tx_param * (self.num + 1); // Kept as var
        if (data_region_hi > nx) {
            data_region_hi = nx;
        }

        self.lo = data_region_lo -% ghosts;
        self.hi = data_region_hi +% ghosts;
        self.sz = self.hi -% self.lo;

        try self.data.resize(self.sz);
        try self.data2.resize(self.sz);

        const off: usize = 1;
        for (0..self.sz) |n_idx| {
            self.data.items[n_idx] = @as(f64, @floatFromInt(self.lo +% n_idx)) + @as(f64, @floatFromInt(off));
            self.data2.items[n_idx] = 0.0;
        }
        return self;
    }

    pub fn deinit(self: *Worker) void {
        self.data.deinit(); // ArrayLists store their allocator
        self.data2.deinit();
    }

    fn threadEntry(w: *Worker) void {
        w.run() catch |err| {
            print("Thread {} run failed: {any}\n", .{ w.num, err });
        };
    }

    pub fn start(self: *Worker) !void {
        // Use default allocator for thread stack (null argument)
        self.thread_handle = try Thread.spawn(self.allocator, Worker.threadEntry, .{self});
    }

    pub fn join(self: *Worker) void {
        if (self.thread_handle) |h| {
            h.join();
            self.thread_handle = null;
        }
    }

    fn recv_ghosts(self: *Worker) void {
        self.data.items[0] = self.left_q.pop();
        self.data.items[self.sz - 1] = self.right_q.pop();
    }

    fn update(self: *Worker) void {
        self.recv_ghosts();

        const c_const: f64 = k * dt / (dx * dx);
        var n: usize = 1;
        while (n < self.sz - 1) : (n += 1) {
            self.data2.items[n] = self.data.items[n] + c_const * (self.data.items[n + 1] + self.data.items[n - 1] - 2.0 * self.data.items[n]);
        }

        std.mem.swap(ArrayList(f64), &self.data, &self.data2);

        self.send_ghosts();
    }

    fn send_ghosts(self: *Worker) void {
        if (self.leftThread) |lt| {
            lt.right_q.push(self.data.items[1]);
        } else {
            print("Warning: leftThread is null for worker {}. This may be an issue.\n", .{self.num});
        }

        if (self.rightThread) |rt| {
            rt.left_q.push(self.data.items[self.sz - 2]);
        } else {
            print("Warning: rightThread is null for worker {}. This may be an issue.\n", .{self.num});
        }
    }

    pub fn run(self: *Worker) !void {
        self.send_ghosts();
        var t: usize = 0;
        while (t < nt) : (t += 1) {
            self.update();
        }
        self.recv_ghosts();
    }
};

fn construct_grid(workers_slice: []const Worker, allocator: Allocator) !ArrayList(f64) {
    var total = ArrayList(f64).init(allocator);
    if (nx == 0) return total; // Return empty if global grid size is 0

    try total.resize(nx); // Sets length, new items are undefined

    for (workers_slice) |*w_const| {
        var current_global_idx: usize = w_const.lo +% ghosts; // Start of actual data for this worker
        var local_idx_n: usize = ghosts; // Index within worker's data array

        while (local_idx_n < w_const.sz -% ghosts) : (local_idx_n += 1) {
            if (current_global_idx < nx) { // Check bounds for total grid
                total.items[current_global_idx] = w_const.data.items[local_idx_n];
            } else {
                // Should not happen if partitioning is correct and nx > 0
                print("Warning: construct_grid index out of bounds. current_global_idx={}, nx={}\n", .{ current_global_idx, nx });
                break;
            }
            current_global_idx += 1;
        }
    }
    return total;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit(); // Ensure GPA is deinitialized
    const allocator = gpa.allocator();

    var workers_list = ArrayList(Worker).init(allocator);
    defer {
        for (workers_list.items) |*w| {
            w.deinit();
        }
        workers_list.deinit();
    }

    var tx: usize = 0;
    if (threads_count > 0) {
        tx = (2 * ghosts + nx) / threads_count;
    } else if (nx > 0) { // If nx > 0 but threads_count is 0, it's problematic
        print("Warning: threads_count is 0, but nx > 0. No work will be done.\n", .{});
        // Potentially set threads_count to 1 or handle as an error
    }

    if (threads_count > 0) {
        var th: usize = 0;
        while (th < threads_count) : (th += 1) {
            try workers_list.append(try Worker.init(allocator, th, tx));
        }

        var th_ptr_setup: usize = 0;
        while (th_ptr_setup < threads_count) : (th_ptr_setup += 1) {
            const next: usize = (th_ptr_setup + 1) % threads_count;
            const prev: usize = (th_ptr_setup + threads_count - 1) % threads_count;

            workers_list.items[th_ptr_setup].rightThread = &workers_list.items[next];
            workers_list.items[th_ptr_setup].leftThread = &workers_list.items[prev];
        }
    }

    const t1 = time.nanoTimestamp();
    if (threads_count > 0) {
        for (workers_list.items) |*w| {
            try w.start();
        }
        for (workers_list.items) |*w| {
            w.join();
        }
    }
    const t2 = time.nanoTimestamp();
    const elapsed_ns = t2 - t1;
    const elapsed_s: f64 = @as(f64, @floatFromInt(elapsed_ns)) * 1e-9;

    try std.io.getStdOut().writer().print("elapsed: {d:e}\n", .{elapsed_s});

    if (nx > 0 and threads_count > 0) {
        var total_grid = try construct_grid(workers_list.constSlice(), allocator);
        defer total_grid.deinit();

        if (nx <= 20) {
            try pr_grid(total_grid.items);
        }
    } else if (nx <= 20) { // If nx is small but no work done, print empty or indication
        try std.io.getStdOut().writer().print("[] (No data generated for grid display)\n", .{});
    }

    const perf_filename = "perfdata.csv";
    var file_existed: bool = true;
    _ = fs.cwd().access(perf_filename, .{}) catch |err| {
        if (err == error.FileNotFound) {
            file_existed = false;
        } else return err; // Propagate other access errors
    };

    if (!file_existed) {
        var header_file = try fs.cwd().createFile(perf_filename, .{});
        // Ensure header_file is closed after writing.
        // The defer here might be tricky if openFile for append fails later.
        // Simpler to close it immediately.
        errdefer header_file.close(); // Close if subsequent operations fail
        try header_file.writer().print("lang,nx,nt,threads,dt,dx,total time,flops\n", .{});
        header_file.close(); // Explicit close after successful write
    }

    var data_file = try fs.cwd().openFile(perf_filename, .{ .mode = .append });
    defer data_file.close();
    try data_file.writer().print("zig,{d},{d},{d},{d:.1},{d:.1},{d:.9},0\n", .{
        nx, nt, threads_count, dt, dx, elapsed_s,
    });
}
