import psutil
import threading
from pydantic import BaseModel
from typing import List


class Report(BaseModel):
    cpu_avg: float
    cpu_max: float
    cpu_p99: float
    cpu_p95: float
    cpu_p90: float
    mem_avg: float
    mem_max: float
    mem_p99: float
    mem_p95: float
    mem_p90: float
    net_sent_avg: float
    net_sent_max: float
    net_sent_p99: float
    net_sent_p95: float
    net_sent_p90: float
    net_recv_avg: float
    net_recv_max: float
    net_recv_p99: float
    net_recv_p95: float
    net_recv_p90: float
    

class MonitorReportor:

    def __init__(self, cpu_data: list, mem_data: list, netio_data: list):
        self.cpu_data = cpu_data
        self.mem_data = mem_data
        self.net_sent_data = [data[0] for data in netio_data]
        self.net_recv_data = [data[1] for data in netio_data]
        

    def get_report(self) -> Report:
        self.cpu_data.sort()
        self.mem_data.sort()
        self.net_sent_data.sort()
        self.net_recv_data.sort()
        data = {
            "cpu_avg": sum(self.cpu_data) / len(self.cpu_data),
            "cpu_max": self.cpu_data[-1],
            "cpu_p99": self.cpu_data[int(len(self.cpu_data) * 0.99)],
            "cpu_p95": self.cpu_data[int(len(self.cpu_data) * 0.95)],
            "cpu_p90": self.cpu_data[int(len(self.cpu_data) * 0.90)],
            "mem_avg": sum(self.mem_data) / len(self.mem_data),
            "mem_max": self.mem_data[-1],
            "mem_p99": self.mem_data[int(len(self.mem_data) * 0.99)],
            "mem_p95": self.mem_data[int(len(self.mem_data) * 0.95)],
            "mem_p90": self.mem_data[int(len(self.mem_data) * 0.90)],
            "net_sent_avg": sum(self.net_sent_data) / len(self.net_sent_data),
            "net_sent_max": self.net_sent_data[-1],
            "net_sent_p99": self.net_sent_data[int(len(self.net_sent_data) * 0.99)],
            "net_sent_p95": self.net_sent_data[int(len(self.net_sent_data) * 0.95)],
            "net_sent_p90": self.net_sent_data[int(len(self.net_sent_data) * 0.90)],
            "net_recv_avg": sum(self.net_recv_data) / len(self.net_recv_data),
            "net_recv_max": self.net_recv_data[-1],
            "net_recv_p99": self.net_recv_data[int(len(self.net_recv_data) * 0.99)],
            "net_recv_p95": self.net_recv_data[int(len(self.net_recv_data) * 0.95)],
            "net_recv_p90": self.net_recv_data[int(len(self.net_recv_data) * 0.90)],
        }
        return Report(**data)

    
    @staticmethod
    def merge_report(report_list: List[Report]) -> dict:
        cpu_avg = 0
        cpu_max = 0
        cpu_p99 = 0
        cpu_p95 = 0
        cpu_p90 = 0
        mem_avg = 0
        mem_max = 0
        mem_p99 = 0
        mem_p95 = 0
        mem_p90 = 0
        net_sent_avg = 0
        net_sent_max = 0
        net_sent_p99 = 0
        net_sent_p95 = 0
        net_sent_p90 = 0
        net_recv_avg = 0
        net_recv_max = 0
        net_recv_p99 = 0
        net_recv_p95 = 0
        net_recv_p90 = 0
        
        for report in report_list:
            cpu_avg += report.cpu_avg
            cpu_max = max(cpu_max, report.cpu_max)
            cpu_p99 = max(cpu_p99, report.cpu_p99)
            cpu_p95 = max(cpu_p95, report.cpu_p95)
            cpu_p90 = max(cpu_p90, report.cpu_p90)
            mem_avg += report.mem_avg
            mem_max = max(mem_max, report.mem_max)
            mem_p99 = max(mem_p99, report.mem_p99)
            mem_p95 = max(mem_p95, report.mem_p95)
            mem_p90 = max(mem_p90, report.mem_p90)
            net_sent_avg += report.net_sent_avg
            net_sent_max = max(net_sent_max, report.net_sent_max)
            net_sent_p99 = max(net_sent_p99, report.net_sent_p99)
            net_sent_p95 = max(net_sent_p95, report.net_sent_p95)
            net_sent_p90 = max(net_sent_p90, report.net_sent_p90)
            net_recv_avg += report.net_recv_avg
            net_recv_max = max(net_recv_max, report.net_recv_max)
            net_recv_p99 = max(net_recv_p99, report.net_recv_p99)
            net_recv_p95 = max(net_recv_p95, report.net_recv_p95)
            net_recv_p90 = max(net_recv_p90, report.net_recv_p90)

        cpu_avg /= len(report_list)
        mem_avg /= len(report_list)
        net_sent_avg /= len(report_list)
        net_recv_avg /= len(report_list)
        return Report(
            cpu_avg=cpu_avg,
            cpu_max=cpu_max,
            cpu_p99=cpu_p99,
            cpu_p95=cpu_p95,
            cpu_p90=cpu_p90,
            mem_avg=mem_avg,
            mem_max=mem_max,
            mem_p99=mem_p99,
            mem_p95=mem_p95,
            mem_p90=mem_p90,
            net_sent_avg=net_sent_avg,
            net_sent_max=net_sent_max,
            net_sent_p99=net_sent_p99,
            net_sent_p95=net_sent_p95,
            net_sent_p90=net_sent_p90,
            net_recv_avg=net_recv_avg,
            net_recv_max=net_recv_max,
            net_recv_p99=net_recv_p99,
            net_recv_p95=net_recv_p95,
            net_recv_p90=net_recv_p90,
        )
        
        
    
    


class ProcessMonitor:
    _monitor = None
    
    def __init__(self, interval=1):
        self.cpu_data = []
        self.mem_data = []
        self.netio_data = []
        self.interval = interval
        self.thread = None
        self.stop = True
        
    def reset(self):
        self.cpu_data = []
        self.mem_data = []
        self.netio_data = []
        self.stop = True
        
        if self.thread is not None:
            self.thread.join()
        self.thread = None
        self.stop = False
        
    
    @staticmethod
    def get_instance():
        if ProcessMonitor._monitor is None:
            ProcessMonitor._monitor = ProcessMonitor()
        return ProcessMonitor._monitor
    
        
    def start_monitor(self):
        self.reset()
        
        def monitor():
            import time
            # 获取当前进程
            process = psutil.Process()
            
            # 获取初始网络统计
            net_io_start = psutil.net_io_counters()

            # 记录初始时间
            start_time = time.time()
            
            while not self.stop:
                time.sleep(self.interval)
                # CPU利用率（按单个进程计算）
                cpu_usage = process.cpu_percent(interval=self.interval)
                
                # 内存占用
                memory_info = process.memory_info()
                memory_usage = memory_info.rss / (1024 ** 2)  # 将 bytes 转为 MB
                
                # 网络速度（上行和下行）
                net_io_end = psutil.net_io_counters()
                now_time = time.time()
                elapsed_time = now_time - start_time

                # 计算网速
                bytes_sent_per_sec = (net_io_end.bytes_sent - net_io_start.bytes_sent) / elapsed_time
                bytes_recv_per_sec = (net_io_end.bytes_recv - net_io_start.bytes_recv) / elapsed_time
                
                # 更新初始状态
                net_io_start = net_io_end
                start_time = time.time()
                
                self.cpu_data.append(cpu_usage)
                self.mem_data.append(memory_usage)
                self.netio_data.append((bytes_sent_per_sec, bytes_recv_per_sec))

        self.thread = threading.Thread(target=monitor)
        self.thread.start()
    
    def stop_monitor(self) -> Report:
        self.stop = True
        self.thread.join()
        return MonitorReportor(self.cpu_data, self.mem_data, self.netio_data).get_report()
    
if __name__ == "__main__":
    monitor = ProcessMonitor()
    monitor.start_monitor()
    tt = 0
    data = []
    for i in range(100000000):
        tt += i
        data.append(tt)
    print(monitor.stop_monitor())