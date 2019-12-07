import multiprocessing
import random
import time

ml = [random.random() for i in range(800)]


def worker(val, send_end):
    result = val * 25
    # print(result)
    send_end.send(result)


def process_mt():
    init = time.time()
    jobs = []
    pipe_list = []
    for val in ml:
        recv_end, send_end = multiprocessing.Pipe(False)
        p = multiprocessing.Process(target=worker, args=(val, send_end))
        jobs.append(p)
        pipe_list.append(recv_end)
        p.start()

    for proc in jobs:
        proc.join()
    result_list = [x.recv() for x in pipe_list]
    print(time.time() - init)


def process():
    init = time.time()

    result_list = []
    for val in ml:
        result_list.append(val * 25)

    print(time.time() - init)


if __name__ == '__main__':
    process_mt()
    process()
