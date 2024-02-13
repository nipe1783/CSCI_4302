import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import matplotlib.pyplot as plt

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(String, 'topic', self.listener_callback, 10)
        self.i = 0
        self.receive_times = []

    def listener_callback(self, msg):
        receive_time = time.time()
        self.receive_times.append(receive_time)
        
        self.i += 1
        self.get_logger().info(f'I heard: "{msg.data}", message number: {self.i}')

        if self.i == 1:
            self.start_time = receive_time

        if self.i >= 400:
            self.end_time = receive_time
            self.calculate_differences_and_plot()
            rclpy.shutdown()

    def calculate_differences_and_plot(self):
        time_differences = [j - i for i, j in zip(self.receive_times[:-1], self.receive_times[1:])]
        
        plt.hist(time_differences, bins=30, edgecolor='black')
        plt.title('Histogram of Publisher Node Wait Time (400 messages)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    try:
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()