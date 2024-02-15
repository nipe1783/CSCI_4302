import rclpy
from rclpy.node import Node
from rclpy.time import Time
from builtin_interfaces.msg import Time as TimeMsg

class SystemClockPublisher(Node):
    def __init__(self):
        super().__init__('system_clock_publisher')
        self.publisher = self.create_publisher(TimeMsg, 'system_clock', 10)
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.publish_current_time)

    def publish_current_time(self):
        now = self.get_clock().now()
        time_msg = TimeMsg()
        time_msg.sec = now.seconds_nanoseconds()[0]
        time_msg.nanosec = now.seconds_nanoseconds()[1]
        self.publisher.publish(time_msg)
        self.get_logger().info(f'Publishing: {time_msg.sec} seconds and {time_msg.nanosec} nanoseconds')

def main(args=None):
    rclpy.init(args=args)
    system_clock_publisher = SystemClockPublisher()
    try:
        rclpy.spin(system_clock_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        system_clock_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()