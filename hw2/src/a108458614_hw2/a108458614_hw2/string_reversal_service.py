import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from a108458614_service.srv import ProcessString
import time

class StringReversalService(Node):

    def __init__(self):
        super().__init__('string_reversal_service')
        self.srv = self.create_service(ProcessString, 'reverse_string', self.reverse_string_callback)

    def reverse_string_callback(self, request, response):
        start_time = time.time()
        response.output_str = request.input[::-1]
        end_time = time.time()
        response.output_duration = end_time - start_time
        self.get_logger().info(f'Received: {request.input}. Sending back: {response.output_str} and {response.output_duration} seconds.')
        return response

def main(args=None):
    rclpy.init(args=args)
    service = StringReversalService()
    rclpy.spin(service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()