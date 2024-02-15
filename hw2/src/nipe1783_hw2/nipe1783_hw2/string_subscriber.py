import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from builtin_interfaces.msg import Time as TimeMsg
import matplotlib.pyplot as plt
import time

class TimeDifferenceSubscriber(Node):
    def __init__(self):
        super().__init__('time_difference_subscriber')
        self.subscription = self.create_subscription(
            TimeMsg,
            'system_clock',
            self.listener_callback,
            10)
        self.clock = Clock()
        self.message_times = []
        self.received_times = []
        self.i = 0

    def listener_callback(self, msg):
        now = self.clock.now()
        print(f"Received: {msg.sec} seconds and {msg.nanosec} nanoseconds")
        now_sec = now.to_msg().sec + now.to_msg().nanosec / 1e9
        msg_time_sec = msg.sec + msg.nanosec / 1e9
        self.message_times.append(msg_time_sec)
        self.received_times.append(now_sec)
        self.i += 1

        if self.i >= 400:
            self.calculate_differences_and_plot()
            rclpy.shutdown()

    def calculate_differences_and_plot(self):
        time_differences = [recv - msg for msg, recv in zip(self.message_times, self.received_times)]
        
        plt.hist(time_differences, bins=30, edgecolor='black')
        plt.title('Histogram of Time Differences for 400 Messages')
        plt.xlabel('Time Difference (seconds)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    print("Starting Subscriber...")
    time_difference_subscriber = TimeDifferenceSubscriber()
    try:
        rclpy.spin(time_difference_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        time_difference_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()