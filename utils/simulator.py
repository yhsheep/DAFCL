import heapq
import numpy as np

class AsyncSimulator:
    def __init__(self, args):
        self.args = args
        self.current_time = 0.0
        # 优先队列：(完成时间, 客户端ID, 更新权重, 开始轮次, 任务ID)
        self.event_queue = [] 
        # [新增] 追踪当前正在计算/传输中的客户端，防止重复激活
        self.in_flight_clients = set() 
        self.processed_events = 0

    def register_event(self, client_id, update_weights, start_round, task_id):
        """
        注册计算完成事件，模拟异步延迟。
        """
        # 记录该客户端为“忙碌”状态
        self.in_flight_clients.add(client_id)
        
        # 延迟模拟 (Exponential 更好地模拟长尾时延)
        if self.args.delay_dist == 'exponential':
            delay = np.random.exponential(scale=self.args.max_delay)
        else:
            delay = np.random.uniform(0.1, self.args.max_delay)
            
        delay = max(0.1, delay) # 避免时间归零
        finish_time = self.current_time + delay
        
        heapq.heappush(self.event_queue, (finish_time, client_id, update_weights, start_round, task_id))
        
    def get_next_event(self):
        if not self.event_queue:
            return None
            
        finish_time, client_id, update_weights, start_round, task_id = heapq.heappop(self.event_queue)
        
        # 更新全局虚拟时钟
        self.current_time = finish_time
        self.processed_events += 1
        
        # 计算结束，客户端恢复空闲
        self.in_flight_clients.remove(client_id)
        
        return client_id, update_weights, start_round, task_id

    def is_client_busy(self, client_id):
        return client_id in self.in_flight_clients

    def empty(self):
        return len(self.event_queue) == 0